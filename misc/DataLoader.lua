require 'hdf5'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  -------------------------------------------------------------------------------
  -- load the json file
  -------------------------------------------------------------------------------
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
  print('vocab size is ' .. self.vocab_size)
  
  -------------------------------------------------------------------------------
  -- open the hdf5 file
  -------------------------------------------------------------------------------
  print('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 4, '/images should be a 4D tensor')
  assert(images_size[3] == images_size[4], 'width and height must match')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.max_image_size = images_size[3]
  print(string.format('read %d images of size %dx%dx%d', self.num_images, 
            self.num_channels, self.max_image_size, self.max_image_size))

  -- load in the sequence data
  local seq_size = self.h5_file:read('/labels'):dataspaceSize()
  self.seq_length = seq_size[2]
  print('max sequence length in data is ' .. self.seq_length)
  -- load the pointers in full to RAM (should be small enough)
  self.label_start_ix = self.h5_file:read('/label_start_ix'):all()
  self.label_end_ix = self.h5_file:read('/label_end_ix'):all()
  
  -------------------------------------------------------------------------------
  -- separate out indexes for each of the provided splits
  -------------------------------------------------------------------------------
  self.split_ix = {}
  self.iterators = {}
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.split_ix[split] then
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') 
  local batch_size = utils.getopt(opt, 'batch_size', 5) 
  local neg_time = opt.neg_time
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 5) 

  local split_ix = self.split_ix[split] 
  assert(split_ix, 'split ' .. split .. ' not found.')
  local max_index = #split_ix
  -------------------------------------------------------------------------------
  -- Load positive samples
  -------------------------------------------------------------------------------
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local label_batch = torch.LongTensor(batch_size * seq_per_img, self.seq_length)
  local seqlen_batch = torch.LongTensor(batch_size * seq_per_img)

  local wrapped = false
  local infos = {}
  for i=1,batch_size do
    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next
    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- fetch the image from h5
    local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            {1,self.max_image_size},{1,self.max_image_size})
    img_batch_raw[i] = img
    -- fetch the sequence labels
    local ix1 = self.label_start_ix[ix]
    local ix2 = self.label_end_ix[ix]
    local ncap = ix2 - ix1 + 1 -- number of captions available for this image
    assert(ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t')

    local seq
    if ncap < seq_per_img then
      -- we need to subsample (with replacement)
      seq = torch.LongTensor(seq_per_img, self.seq_length)
      for q=1, seq_per_img do
        local ixl = torch.random(ix1,ix2)
        seq[{ {q,q} }] = self.h5_file:read('/labels'):partial({ixl, ixl}, {1,self.seq_length})
      end
    else
      -- there is enough data to read a contiguous chunk, but subsample the chunk position
      local ixl = torch.random(ix1, ix2 - seq_per_img + 1) -- generates integer in the range
      seq = self.h5_file:read('/labels'):partial({ixl, ixl+seq_per_img-1}, {1,self.seq_length})
    end

    local il = (i-1)*seq_per_img+1
    label_batch[{ {il,il+seq_per_img-1} }] = seq
    seqlen_batch[{ {il,il+seq_per_img-1} }] = torch.sum(seq:ne(0), 2)

    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.info.images[ix].id
    info_struct.file_path = self.info.images[ix].file_path
    table.insert(infos, info_struct)
  end

  local data = {}
  data.images = img_batch_raw
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.seqlen = seqlen_batch
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos
  -------------------------------------------------------------------------------
  -- If on test mode, break from here
  -------------------------------------------------------------------------------
  if split == 'test' then
    return data
  end
  -------------------------------------------------------------------------------
  -- Load negative samples
  -------------------------------------------------------------------------------
  local neg_batch_size = batch_size*neg_time
  local img_batch_raw_neg = torch.ByteTensor(neg_batch_size, 3, 256, 256)
  local label_batch_neg = torch.LongTensor(neg_batch_size, self.seq_length)
  local seqlen_batch_neg = torch.LongTensor(neg_batch_size)

  for i=1,neg_batch_size do
    local ri = torch.randperm(max_index)
    local randtxt = split_ix[ri[1]]
    local randimg = split_ix[ri[2]]

    -- fetch the image from h5
    local img = self.h5_file:read('/images'):partial({randimg,randimg},{1,self.num_channels},
                            {1,self.max_image_size},{1,self.max_image_size})
    img_batch_raw_neg[i] = img
    -- fetch the sequence labels
    local ix1 = self.label_start_ix[randtxt]
    local ix2 = self.label_end_ix[randtxt]
    local ncap = ix2 - ix1 + 1 -- number of captions available for this image
    assert(ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t')

    local seq
    if ncap < seq_per_img then
      -- we need to subsample (with replacement)
      seq = torch.LongTensor(seq_per_img, self.seq_length)
      for q=1, seq_per_img do
        local ixl = torch.random(ix1,ix2)
        seq[{ {q,q} }] = self.h5_file:read('/labels'):partial({ixl, ixl}, {1,self.seq_length})
      end
    else
      -- there is enough data to read a contiguous chunk, but subsample the chunk position
      local ixl = torch.random(ix1, ix2 - seq_per_img + 1) -- generates integer in the range
      seq = self.h5_file:read('/labels'):partial({ixl, ixl+seq_per_img-1}, {1,self.seq_length})
    end

    local il = (i-1)*seq_per_img+1
    label_batch_neg[{ {il,il+seq_per_img-1} }] = seq
    seqlen_batch_neg[{ {il,il+seq_per_img-1} }] = torch.sum(seq:ne(0), 2)
  end

  local data_neg = {}
  data_neg.images = img_batch_raw_neg
  data_neg.labels = label_batch_neg:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data_neg.seqlen = seqlen_batch_neg
  -------------------------------------------------------------------------------
  -- Put all the data together
  -------------------------------------------------------------------------------
  local all_batch_size = batch_size + neg_batch_size
  local img_all = torch.ByteTensor(all_batch_size, 3, 256, 256)
  local txt_all = torch.LongTensor(self.seq_length, all_batch_size)
  local seqlen_all = torch.LongTensor(all_batch_size)  
  local label_all = torch.LongTensor(all_batch_size):fill(1)

  img_all:narrow(1,1,batch_size):copy(data.images)
  txt_all:narrow(2,1,batch_size):copy(data.labels)
  seqlen_all:narrow(1,1,batch_size):copy(data.seqlen)

  img_all:narrow(1,batch_size+1, neg_batch_size):copy(data_neg.images)
  txt_all:narrow(2,batch_size+1, neg_batch_size):copy(data_neg.labels)
  seqlen_all:narrow(1,batch_size+1, neg_batch_size):copy(data_neg.seqlen)
  label_all:narrow(1,batch_size+1, neg_batch_size):copy(torch.Tensor(neg_batch_size):zero())

  -------------------------------------------------------------------------------
  -- shuffle all the data
  -------------------------------------------------------------------------------
  local ri = torch.randperm(all_batch_size)
  local imgall = torch.ByteTensor(all_batch_size, 3, 256, 256)
  local txtall = torch.LongTensor(self.seq_length, all_batch_size)
  local seqlenall = torch.LongTensor(all_batch_size)
  local labelall = torch.LongTensor(all_batch_size)
  for i=1,all_batch_size do
    imgall[{i, {}, {}, {}}] = img_all[{ri[i], {}, {}, {}}]
    txtall[{{}, i}] = txt_all[{{}, ri[i]}]
    seqlenall[i] = seqlen_all[ri[i]]
    labelall[i] = label_all[ri[i]]
  end

  local datall = {}
  datall.images = imgall
  datall.labels = txtall
  datall.seqlen = seqlenall
  datall.cls = labelall
  datall.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  datall.infos = infos

  return datall
end


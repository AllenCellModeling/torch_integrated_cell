require 'hdf5'
require 'nn'
require 'paths'
require 'torchx'
require 'utils'

local DataProvider = {
    image_paths = nil, -- Size of isotropic multivariate Gaussian Z
    labels = nil,
    label_names = nil,
    opts = nil
}

function DataProvider.create(im_parent, opts)
    local self = {}
    self.opts = {}
    
    -- set the random seed
    self.opts.seed = opts.seed or 1
    local gen = torch.Generator()
    torch.manualSeed(gen, self.opts.seed)
    torch.random(gen)
    
    --     shallow copy these options
    self.opts.channel_inds_in = opts.channel_inds_in:clone() 
    self.opts.channel_inds_out = opts.channel_inds_out:clone()
    -- self.opts.rotate = opts.rotate
   
--  search the image directory for files

    local image_paths = {}
    local classes = {}
    
    local c = 0
    for dir in paths.iterdirs(im_parent) do
        local im_dir = im_parent .. '/' .. dir
        for f in paths.files(im_dir, '.h5') do
            c = c+1
            image_paths[c] = im_dir .. '/' .. f
            classes[c] = dir
        end    
    end
    
    self.image_paths = image_paths;
    local label_names, labels = utils.unique(classes)
    self.label_names = label_names
        
    local nImgs = #image_paths
    local nClasses = torch.max(labels)
    local one_hot = torch.zeros(nImgs, nClasses):long()
    for i = 1,nImgs do
        one_hot[{i,labels[i]}] = 1
    end

    -- save 5% of the data for testing
    local rand_inds = torch.randperm(gen, nImgs):long()
    local nTest = torch.round(nImgs/20)
    
    self.train = {}
    self.train.inds = rand_inds[{{nTest+1,-1}}]
    self.train.labels = one_hot:index(1, self.train.inds)
    
    self.test = {}
    self.test.inds = rand_inds[{{1,nTest}}]
    self.test.labels = one_hot:index(1, self.test.inds)
    
    self.image_paths = image_paths
    
--     I'm not sure what this does but I'm pretty sure it has to do with objectifying the 'self'
    setmetatable(self, { __index = DataProvider })    
    return self

end

function DataProvider:getImages(indices, train_or_test)
    
    local images_in, images_out = {}, {}
        
    c = 1
    for ind = 1,indices:numel() do
        local i = indices[ind]
        
        -- print('Reading image ' .. ind .. '/' .. indices:numel() .. ': ' .. self.image_paths[self[train_or_test].inds[i]])
        
        local im_path = self.image_paths[self[train_or_test].inds[i]]
        
        local worked, im = pcall(self.loadImage, im_path)
        
        if worked then
            -- local im = self.loadImage(im_path)
            local im = nn.Unsqueeze(1):forward(im)

            --         if self.opts.flip == true then
            --             if torch.rand(1) > 0.5 then
            --                 im = image.flip(im, 4)
            --             end

            --             if torch.rand(1) > 0.5 then
            --                 im = image.flip(im, 5)
            --             end
            --         end

            images_in[c] = im:index(2, self.opts.channel_inds_in)
            images_out[c] = im:index(2, self.opts.channel_inds_out)
            c = c+1
        else
            print('Failed to load image ' .. im_path)
        end
    end
    
    local images_in = torch.concat(images_in,1)
    local images_out = torch.concat(images_out,1)
    
    return images_in, images_out
end

function DataProvider:getLabels(indices, train_or_test)
    local labels_in = self[train_or_test].labels:index(1, indices):typeAs(self.test.labels)
    return labels_in
end


function DataProvider.loadImage(hdf5_path)
--    print(hdf5_path)
    local myFile = hdf5.open(hdf5_path, 'r')
    local data = myFile:read():all()

    local im_nuc = torch.Tensor(data.im_nuc:size()):copy(data.im_nuc)
    local im_cell = torch.Tensor(data.im_cell:size()):copy(data.im_cell)
    local im_memb = torch.Tensor(data.im_memb:size()):copy(data.im_memb)
    local im_struct = torch.Tensor(data.im_struct:size()):copy(data.im_struct)
    local im_dna = torch.Tensor(data.im_dna:size()):copy(data.im_dna)
    
    data = nil

    im_nuc = nn.Unsqueeze(1):forward(im_nuc)    
    im_cell = nn.Unsqueeze(1):forward(im_cell)
    im_dna = nn.Unsqueeze(1):forward(im_dna)
    im_memb = nn.Unsqueeze(1):forward(im_memb)
    im_struct = nn.Unsqueeze(1):forward(im_struct)
    
    
--     function imscale(img, mask)
--         m = prctile(img[torch.gt(mask,0)], 5)
--         img = torch.add(img, -m)
--         img[torch.lt(img,0)] = 0
--         img = torch.div(img, torch.max(img))
--         return img
--     end

--     function prctile(dat, prct)
--         svals = torch.sort(dat:view(dat:nElement()))
--         len = svals:size(1)
--         return svals[torch.round(len*prct*1E-2)]
--     end
    
--     im_dna = imscale(im_dna, im_nuc)
--     im_memb = imscale(im_memb, im_nuc)
--     im_struct = imscale(im_struct, im_nuc)    

    local im = torch.cat({im_dna, im_memb, im_struct, im_nuc, im_cell}, 1)
    im = torch.div(im, torch.max(im))
    im = im:permute(1,2,4,3)
    return im 
end

return DataProvider



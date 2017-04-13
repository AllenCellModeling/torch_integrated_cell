require 'nn'
require 'torchx'
require 'imtools'

local DataProvider = {
    data = nil, -- Size of isotropic multivariate Gaussian Z
    labels = nil,
    opts = nil
}

function DataProvider.create(imageDir, opts)
    
    dataSavePath = opts.save.data
    
    local data
    local labels
    local data_info
    
    local data = {}
    if paths.filep(dataSavePath) then
        print('Loading data from ' .. dataSavePath)
        data = torch.load(dataSavePath)
        
        print('Done')
    else
        
        local c = 0
        local images, image_paths, classes = {}, {}, {}
        
        dirs = {}
        for dir in paths.iterdirs(imageDir) do
            dirs[#dirs+1] = dir
        end
        
        dirs = utils.alphanumsort(dirs)
        
        for i = 1,#dirs do
            dir = dirs[i]
            print('Loading images from ' .. imageDir .. '/' .. dir)

            local images_tmp, image_paths_tmp = imtools.load_img(imageDir .. '/' .. dir .. '/', 'png', opts.image_sub_size)

            for i = 1,#image_paths_tmp do
                c = c+1
                
                images[c] = nn.Unsqueeze(1):forward(images_tmp[i])
                image_paths[c] = image_paths_tmp[i]
            
                tokens = utils.split(image_paths_tmp[i], '/')
                classes[c] = tokens[#tokens-1]    
            end
        end
        images = torch.concat(images,1)
        images = torch.FloatTensor(images:size()):copy(images)
        classes, labels = utils.unique(classes)
        
        local nImgs = images:size()[1]
        local nClasses = torch.max(labels)
        local one_hot = torch.zeros(nImgs, nClasses):long()
        for i = 1,nImgs do
            one_hot[{i,labels[i]}] = 1
        end

        -- save 5% of the data for testing
        local rand_inds = torch.randperm(nImgs):long()
        local nTest = torch.round(nImgs/20)

        data.train = {}
        data.train.inds = rand_inds[{{nTest+1,-1}}]
        data.train.images = images:index(1, data.train.inds)
        data.train.labels = one_hot:index(1, data.train.inds)

        data.test = {}
        data.test.inds = rand_inds[{{1,nTest}}]
        data.test.images = images:index(1, data.test.inds)
        data.test.labels = one_hot:index(1, data.test.inds)

        data.image_paths = image_paths
        data.classes = classes
        
        torch.save(dataSavePath, data)
    end

    local self = data
   
    self.opts = {}
    
--     shallow copy these options
    self.opts.channel_inds_in = opts.channel_inds_in:clone() or torch.LongTensor{1}
    self.opts.channel_inds_out = opts.channel_inds_out:clone() or torch.LongTensor{1}
    self.opts.rotate = opts.rotate or false
    
    setmetatable(self, { __index = DataProvider })    
    return self

end

function DataProvider:getImages(indices, train_or_test)
    
    local images_in = self[train_or_test].images:index(1, indices):index(2, self.opts.channel_inds_in):clone()
    local images_out = self[train_or_test].images:index(1, indices):index(2, self.opts.channel_inds_out):clone()
    
    if self.opts.rotate and torch.rand(1)[1] < 0.01 then            
        for i = 1,images_in:size()[1] do
            rad = (torch.rand(1)*2*math.pi)[1]
            flip = torch.rand(1)[1]>0.5
            if flip then
                images_in[i] = image.hflip(images_in[i])
                images_out[i] = image.hflip(images_out[i])
            end

            images_in[i] = image.rotate(images_in[i], rad)
            images_out[i] = image.rotate(images_out[i], rad)
        end
    end

    return images_in, images_out
end

function DataProvider:getLabels(indices, train_or_test)
    local labels_in = self[train_or_test].labels:index(1, indices):typeAs(self.test.labels)
    return labels_in
end

return DataProvider



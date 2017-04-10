require 'cutorch' -- Use CUDA if available
require 'cudnn'
require 'cunn'
require 'paths'
require 'imtools'

optim = require 'optim'
gnuplot = require 'gnuplot'
image = require 'image'
nn = require 'nn'
optnet = require 'optnet'
gnuplot = require 'gnuplot'

cuda = true
hasCudnn = true

print('Has cudnn: ')
print(hasCudnn)

learner = require 'train'
setup = require 'setup'

unpack = table.unpack or unpack

count = cutorch.getDeviceCount()
--     torch.setdefaulttensortype('torch.CudaTensor')    
print('GPU is ON')
for i = 1, count do
    print('Device ' .. i .. ':')
    freeMemory, totalMemory = cutorch.getMemoryUsage(i)
    print('\t Free memory ' .. freeMemory)
    print('\t Total memory ' .. totalMemory)
end

cmd = torch.CmdLine()
--model settings
cmd:option('-modelName',        'caae-nopool_GAN_v2', 'model name')
cmd:option('-nLatentDims',       16, 'dimensionality of the latent space')
cmd:option('-seed',              1, 'random seed')

-- saving settings
cmd:option('-saveDir',           'results', 'save dir')
cmd:option('-saveStateIter',     25, 'Iterations between saving model states')
cmd:option('-saveProgressIter',  5, 'Iterations between save progress states')

-- gpu settings
cmd:option('-gpu1',              1, 'gpu for the encoder')
cmd:option('-gpu2',              2, 'gpu for the encoder')
cmd:option('-gpu3',              3, 'gpu for the encoder')
cmd:option('-gpu',               -1, 'gpu for the everything(overrides other gpu settings')

-- data settings
cmd:option('-imdir',             '/root/images/release_4_1_17/release_v2/aligned/2D', 'parent directory for images')
cmd:option('-imsize',             256, 'desired size of images in dataset')
cmd:option('-dataDir', './data/','data save dir')
cmd:option('-dataProvider',      'DataProvider2D', 'data provider object')
cmd:option('-evalImageFunc',        'evalImage2D', 'image evaluation function')

-- training settings
cmd:option('-optimizer',         'adam', 'optimization method')
cmd:option('-learnrate',         0.0002, 'learning rate')
cmd:option('-advGenRatio',       1E-4, 'ratio for advGen update')
cmd:option('-advLatentRatio',    1E-4, 'ratio for advLatent update')
cmd:option('-ganNoise',          0.01, 'injection noise for the GAN')
cmd:option('-ganNoiseAllLayers', false, 'add noise on all GAN layers')
cmd:option('-nepochs',           150, 'number of epochs')
cmd:option('-nepochspt2',        300, 'number of epochs for pt2')
cmd:option('-skipGanD',          false, 'use a GAN on the decoder')
cmd:option('-beta1',             0.5, 'beta1 parameter for ADAM descent')
cmd:option('-ndat',              -1, 'number of training data to use')
cmd:option('-batchSize',         32, 'batch size')
cmd:option('-learningRateDecay', 0.999, 'learning rate decay')

-- display settings
cmd:option('-verbose', false, 'verbosity setting')

opts = cmd:parse(arg)

if opts.nepochs < opts.saveStateIter then
    opts.saveStateIter = opts.nepochs
end

if opts.nepochs < opts.saveProgressIter then
    opts.saveProgressIter = opts.nepochs
end

DataProvider = require(opts.dataProvider)
require(opts.evalImageFunc)

if opts.gpu ~= -1 then
    opts.gpu1 = opts.gpu
    opts.gpu2 = opts.gpu
    opts.gpu3 = opts.gpu
end
print('Setting default GPU to ' .. opts.gpu1)
cutorch.setDevice(opts.gpu1)
torch.setnumthreads(12)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opts.seed) 
cutorch.manualSeed(opts.seed)

-- Set up Torch
print('Setting up')

opts = setup.init(opts)
print(opts)

evalIm = function(x_in, x_out)
    local xHat, latentHat, latentHat_var = decoder:forward(encoder:forward(x_in:cuda()))
        
    xHat = imtools.mat2img(xHat)
    x_out = imtools.mat2img(x_out)
        
    -- Plot reconstructions
    recon = torch.cat(image.toDisplayTensor(x_out, 1, x_in:size(1)), image.toDisplayTensor(xHat, 1, x_in:size(1)), 2)

    return recon
end


opts.channel_inds_in = torch.LongTensor{1,3}
opts.channel_inds_out = torch.LongTensor{1,3}
opts.nChOut = opts.channel_inds_in:size(1)
opts.nChIn = opts.channel_inds_out:size(1)

dataProvider = DataProvider.create(opts.imdir, opts.save.data, opts)
x = dataProvider:getImages(torch.LongTensor{1}, 'train')
print(x:size())


opts.nClasses = 0
opts.nOther = 0

criterion = nn.BCECriterion():cuda()

if opts.ndat == -1 then
    ndat = dataProvider.train.inds:size()[1]
else
    ndat = opts.ndat
end

setup.getModel(opts)
if opts.epoch ~= opts.nepochs then  
    learner.loop(ndat)

    decoder = nil  
    adversary = nil
    adversaryGen = nil
end

shapeEncoder = encoder
shapeEncoder:evaluate()

for i = 1,#shapeEncoder.modules do
    shapeEncoder.modules[i]:clearState()
end

collectgarbage()

shapeDataProvider = dataProvider

shape_embeddings = torch.load(opts.save.tmpEmbeddings)


opts.saveDir = opts.saveDir .. '_pattern'
opts = setup.init(opts)

opts.channel_inds_in = torch.LongTensor{1,2,3}
opts.channel_inds_out = torch.LongTensor{1,2,3}
opts.nChOut = opts.channel_inds_in:size(1)
opts.nChIn = opts.channel_inds_out:size(1)

opts.nClasses = shapeDataProvider:getLabels(torch.LongTensor{1}, 'train'):size()[2]
opts.nOther = opts.nLatentDims

criterion = criterion_label
opts.nepochs = opts.nepochspt2

dataProvider = DataProvider.create(opts.imdir, opts.save.data, opts)
function dataProvider:getCodes(indices, train_or_test)
    local codes = shape_embeddings[train_or_test]:index(1, indices):cuda()
    return codes
end

setup.getModel(opts)
if opts.epoch ~= opts.nepochs then  
    learner.loop(ndat)
end

encoder:evaluate()
decoder:evaluate()

print_images = function(encoder, decoder, dataProvider, train_or_test, save_dir)
        
    require 'paths'

    paths.mkdir(save_dir)

    ndat = dataProvider[train_or_test].labels:size(1)
    nclasses = dataProvider[train_or_test].labels:size(2)

    for j = 1,ndat do
--         print('Printing image ' .. j)
        img_num = torch.LongTensor{j}

        local label = dataProvider:getLabels(torch.LongTensor{j,j}, train_or_test)
        local img = dataProvider:getImages(torch.LongTensor{j,j}, train_or_test):cuda()
        local imsize = img:size()
        local out_latent = nil
        out_latent = encoder:forward(img)

        out_latent[3] = torch.zeros(2,opts.nLatentDims):cuda()

        local out_img = torch.Tensor(nclasses, 3, imsize[3], imsize[4]):cuda()

        im_path = dataProvider.image_paths[dataProvider[train_or_test].inds[j]]
        tokens = utils.split(im_path, '/|.')
        im_class = tokens[7]
        im_id = tokens[8]
        
        print('Printing images for ' .. im_class .. ' ' .. im_id)
        
        save_path = save_dir .. '/' .. im_class .. '_'.. im_id .. '_orig.png'
        image.save(save_path, img[1])
        
        for i = 1,#dataProvider.classes do
            save_path = save_dir .. '/' .. im_class .. '_'.. im_id .. '_pred_' .. dataProvider.classes[i] .. '.png'
            
            if not paths.filep(save_path) then
                one_hot = torch.ones(2,nclasses):fill(-25):cuda()
                one_hot[{{1},{i}}] = 0

                out_latent[1] = one_hot
                out_img = decoder:forward(out_latent)[1][2]


                image.save(save_path, out_img)
            end
            
        end
    end
end

print_images(encoder, decoder, dataProvider, 'train', opts.saveDir .. '/' .. 'pred_train')
print_images(encoder, decoder, dataProvider, 'test', opts.saveDir .. '/' .. 'pred_test')
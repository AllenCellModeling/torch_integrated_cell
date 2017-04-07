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

-- package.loaded['modelTools'] = nil
-- require 'modelTools'


DataProvider = require('DataProvider2D')
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
cmd:option('-imsize', 256, 'desired size of images in dataset')
cmd:option('-gpu1', 1, 'gpu for the encoder')
cmd:option('-gpu2', 2, 'gpu for the encoder')
cmd:option('-gpu3', 3, 'gpu for the encoder')
cmd:option('-gpu', -1, 'gpu for the everything(overrides other gpu settings')
cmd:option('-imdir', '/root/images/release_4_1_17/release_v2/aligned/2D', 'parent directory for images')
cmd:option('-learnrate', 0.0002, 'learning rate')
cmd:option('-advgenratio', 1E-4, 'ratio for advGen update')
cmd:option('-advlatentratio', 1E-4, 'ratio for advLatent update')
cmd:option('-suffix', '', 'string suffix')
cmd:option('-ganNoise', 0.01, 'injection noise for the GAN')
cmd:option('-ganNoiseAllLayers', false, 'add noise on all GAN layers')
cmd:option('-nepochs', 150, 'number of epochs')
cmd:option('-nepochspt2', 300, 'number of epochs for pt2')
cmd:option('-useGanD', 1, 'use a GAN on the decoder')
cmd:option('-beta1', 0.5, 'beta1 parameter for ADAM descent')
cmd:option('-ndat', -1, 'number of training data to use')

params = cmd:parse(arg)
params.useGanD = params.useGanD > 0

print(params)

if params.gpu == -1 then
    gpu1 = params.gpu1
    gpu2 = params.gpu2
    gpu3 = params.gpu3
else
    gpu1 = params.gpu
    gpu2 = params.gpu
    gpu3 = params.gpu
end
print('Setting default GPU to ' .. gpu1)

cutorch.setDevice(gpu1)
torch.setnumthreads(12)

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1) 
cutorch.manualSeed(torch.random())




evalIm = function(x_in, x_out)
    local xHat, latentHat, latentHat_var = decoder:forward(encoder:forward(x_in:cuda()))
        
    xHat = imtools.mat2img(xHat)
    x_out = imtools.mat2img(x_out)
        
    -- Plot reconstructions
    recon = torch.cat(image.toDisplayTensor(x_out, 1, 10), image.toDisplayTensor(xHat, 1, 10), 2)

    return recon
end

-- Now setup the probabilstic autoencoder


imsize = params.imsize

model_opts = setup.getModelOpts()
model_opts.parent_dir = 'trainAAE2D_' .. imsize
model_opts.model_name = 'caae-nopool_GAN_v2'
model_opts.save_dir = model_opts.parent_dir .. '/' .. model_opts.model_name .. '_AAEGAN' .. params.suffix
model_opts.image_dir = params.imdir
model_opts.ganNoise = params.ganNoise
model_opts.useGanD = params.useGanD

model_opts.verbose = false

model_opts.imsize = imsize
model_opts.image_sub_size = 1.4707*(512/imsize)

print(model_opts.save_dir)

setup.getLearnOpts(model_opts)
model_opts.rotate = false

dataProvider = DataProvider.create(model_opts.image_dir, model_opts)

x = dataProvider:getImages(torch.LongTensor{1}, 'train')
print(x:size())


opt.batchSize = 32
opt.nepochs = params.nepochs

nLatentDims = 16
model_opts.nLatentDims = nLatentDims


optEnc.learningRate = params.learnrate
optEnc.beta1 = params.beta1
optDec.learningRate = params.learnrate
optDec.beta1 = params.beta1
optAdvGen.learningRate = params.learnrate
optAdvGen.beta1 = params.beta1
optAdv.learningRate = params.learnrate
optAdv.beta1 = params.beta1

model_opts.advGenRatio = params.advgenratio
model_opts.advLatentRatio = params.advlatentratio

opt.saveStateIter = 25
opt.saveProgressIter = 5

criterion = nn.BCECriterion():cuda()


if params.ndat == -1 then
    ndat = dataProvider.train.inds:size()[1]
else
    ndat = params.ndat
end

if opt.epoch ~= opt.nepochs then  

    setup.getModel()
    learner.loop(ndat)

    decoder = nil  
    adversary = nil
    adversaryGen = nil
end


-- -- Now setup the conditional probabilstic autoencoder

setup.getModel()


shapeEncoder = encoder
shapeEncoder:evaluate()


for i = 1,#shapeEncoder.modules do
    shapeEncoder.modules[i]:clearState()
end

collectgarbage()

shapeDataProvider = dataProvider

model_opts.channel_inds_in = torch.LongTensor{1,2,3}
model_opts.channel_inds_out = torch.LongTensor{1,2,3}
model_opts.nChOut = 3
model_opts.nChIn = 3

embedding_file = model_opts.save_dir .. '/progress_embeddings.t7'
model_opts.save_dir = model_opts.parent_dir .. '/' .. model_opts.model_name .. '_AAEGAN' .. params.suffix .. '_pattern'
print(model_opts.save_dir)

model_opts.nLatentDims = nLatentDims
model_opts.nClasses = shapeDataProvider:getLabels(torch.LongTensor{1}, 'train'):size()[2]
model_opts.nOther = nLatentDims


setup.getLearnOpts(model_opts)


optEnc.learningRate = params.learnrate
optEnc.beta1 = params.beta1
optDec.learningRate = params.learnrate
optDec.beta1 = params.beta1
optAdvGen.learningRate = params.learnrate
optAdvGen.beta1 = params.beta1
optAdv.learningRate = params.learnrate
optAdv.beta1 = params.beta1



model_opts.advGenRatio = params.advgenratio
model_opts.advLatentRatio = params.advlatentratio


criterion = criterion_label

opt.saveStateIter = 25
opt.nepochs = params.nepochspt2


shape_embeddings = torch.load(embedding_file)


dataProvider = DataProvider.create(model_opts.image_dir, model_opts)
function dataProvider:getCodes(indices, train_or_test)
    local codes = shape_embeddings[train_or_test]:index(1, indices):cuda()
    return codes
end
setup.getModel()

if opt.epoch ~= opt.nepochs then  
    
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

        out_latent[3] = torch.zeros(2,model_opts.nLatentDims):cuda()

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

print_images(encoder, decoder, dataProvider, 'train', model_opts.save_dir .. '/' .. 'pred_train')
print_images(encoder, decoder, dataProvider, 'test', model_opts.save_dir .. '/' .. 'pred_test')
require 'cutorch' -- Use CUDA if available
require 'cudnn'
require 'cunn'
require 'torchx'

unpack = table.unpack or unpack

-- print(cudnn)

-- Load dependencies
optim = require 'optim'
gnuplot = require 'gnuplot'
image = require 'image'
nn = require 'nn'

require 'paths'
require 'imtools'
optnet = require 'optnet'

count = cutorch.getDeviceCount()
--     torch.setdefaulttensortype('torch.CudaTensor')    
print('GPU is ON')
for i = 1, count do
    print('Device ' .. i .. ':')
    freeMemory, totalMemory = cutorch.getMemoryUsage(i)
    print('\t Free memory ' .. freeMemory)
    print('\t Total memory ' .. totalMemory)
end

gpu1 = 1
gpu2 = 2
gpu3 = 3

cutorch.setDevice(gpu1)
torch.setnumthreads(12)

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1) 
cutorch.manualSeed(torch.random())
-- end

cuda = true
hasCudnn = true

print('Has cudnn: ')
print(hasCudnn)

package.loaded['modelTools'] = nil
require 'modelTools'

package.loaded['./DataProvider3Dv3'] = nil
DataProvider = require ('./DataProvider3Dv3')

package.loaded['trainAAEGANv2'] = nil
learner = require 'trainAAEGANv2'

package.loaded['setupAAEGANv2'] = nil
setup = require 'setupAAEGANv2'

require 'hdf5'

evalIm = function(x_in, x_out)
    local codes = encoder:forward(x_in:cuda())
    local xHat = decoder:forward(codes)

    local myFile = hdf5.open(model_opts.save_dir .. '/progress.h5', 'w')
    myFile:write('/x_out', x_out:float())
    myFile:write('/xHat', xHat:float())
    myFile:close()
    
    xHat = imtools.im2projection(xHat:float())
    dat_out = imtools.im2projection(x_out:float())

    -- Plot reconstructions
    recon = torch.cat(image.toDisplayTensor(dat_out, 1, dat_out:size()[1]), image.toDisplayTensor(xHat, 1, xHat:size()[1]), 2)
    
    return recon
end



-- Now setup the probabilstic autoencoder

learnrate = 0.01
nLatentDims = 128


model_opts = setup.getModelOpts()
model_opts.parent_dir = 'trainAAEGAN3D'
model_opts.model_name = 'caae3D-nopool_GAN_v2'
model_opts.save_dir = model_opts.parent_dir .. '/' .. model_opts.model_name .. '_release_GAN' .. '_' .. nLatentDims
-- model_opts.image_dir = '/root/images/2016_11_08_Nuc_Cell_Seg_8_cell_lines_V22/processed_aligned_hdf5v2/3D/'
model_opts.image_dir = '/root/images/release_4_1_17_v2/'

model_opts.channel_inds_in = torch.LongTensor{1,2}
model_opts.channel_inds_out = torch.LongTensor{1,2}
model_opts.fullConv = true

model_opts.verbose = false


print(model_opts.save_dir)

setup.getLearnOpts(model_opts)
model_opts.rotate = false

dataProvider = DataProvider.create(model_opts.image_dir, model_opts)


x = dataProvider:getImages(torch.LongTensor{1}, 'train')
print(x:size())


model_opts.nLatentDims = nLatentDims

opt.saveProgressIter = plotOnIter

gnuplot = require 'gnuplot'


-- -- these work but slowly
-- optEnc.learningRate = 0.0002
-- optDec.learningRate = 0.0002
-- optAdvGen.learningRate = 0.0002
-- optAdv.learningRate = 0.0002
-- model_opts.advGenRatio = 1E-4
-- model_opts.advLatentRatio = 1E-4

-- -- -- these also work with noise of 0.05
optEnc.learningRate = 0.002
optDec.learningRate = 0.002
optAdvGen.learningRate = 0.002
optAdv.learningRate = 0.002
model_opts.advGenRatio = 1 -- 0.053 @ 50 epochs
model_opts.advLatentRatio = 1E-4

-- -- these also work with noise of 0.05
-- optEnc.learningRate = 0.02
-- optDec.learningRate = 0.02
-- optAdvGen.learningRate = 0.02
-- optAdv.learningRate = 0.02
-- model_opts.advGenRatio = 1 
-- model_opts.advLatentRatio = 1E-4


opt.update_thresh = 0.597
opt.downdate_thresh = 0.69
opt.deltaL1 = 0.0001

opt.saveStateIter = 25
opt.saveProgressIter = 5
opt.batchSize = 32


-- gpu1 = 2
-- gpu2 = 3

-- -- print(gpuIDs)
-- set_gpu_id = function(model, gpuIDs)
--     for i = 1, #model.modules do
--         model.modules[i].device = gpuIDs[i]
--         if i == #model.modules then
--             model.modules[i].outdevice = gpu1
--         else
--             model.modules[i].outdevice = gpuIDs[i+1]
--         end
--     end
-- end

-- gpuIDs = torch.zeros(#encoder.modules):fill(gpu1)
-- gpuIDs:sub(4,#encoder.modules):fill(gpu2)
-- set_gpu_id(encoder, gpuIDs)

-- gpuIDs = torch.zeros(#decoder.modules):fill(gpu2)
-- set_gpu_id(decoder, gpuIDs)

-- gpuIDs = torch.zeros(#adversary.modules):fill(gpu2)
-- set_gpu_id(adversary, gpuIDs)

-- gpuIDs = torch.zeros(#adversaryGen.modules):fill(gpu2)
-- set_gpu_id(adversaryGen, gpuIDs)


criterion = nn.BCECriterion():cuda()

epoch_tmp = 1000
if epoch_tmp >= opt.epoch then
    print('Training autoencoder')
    
    opt.nepochs = opt.epoch + opt.saveStateIter
    
    if opt.epoch ~= opt.nepochs then  
        setup.getModel()        
            
        learner.loop()

        -- clear all of the big variables from memory
        encoder = nil  
        decoder = nil  
        adversary = nil
        theta, gradTheta, thetaAdv, gradThetaAdv = nil, nil, nil, nil
        os.exit()
    end
end


-- print('Done training autoencoder')


-- -- Now setup the conditional probabilstic autoencoder

-- setup.getModel()

-- adversary = nil
-- decoder = nil
-- -- autoencoder = nil


-- collectgarbage()

-- shapeDataProvider = dataProvider

-- model_opts.channel_inds_in = torch.LongTensor{1,2,3}
-- model_opts.channel_inds_out = torch.LongTensor{1,2,3}
-- model_opts.nChOut = 3
-- model_opts.nChIn = 3

-- embedding_file = model_opts.save_dir .. '/progress_embeddings.t7'

-- model_opts.save_dir = model_opts.parent_dir .. '/' .. model_opts.model_name .. '_v2' .. '_pattern'
-- print(model_opts.save_dir)

-- model_opts.nLatentDims = nLatentDims
-- model_opts.nClasses = shapeDataProvider:getLabels(torch.LongTensor{1}, 'train'):size()[2]
-- model_opts.nOther = nLatentDims


-- setup.getLearnOpts(model_opts)


-- opt.update_thresh = 0.597
-- opt.downdate_thresh = 0.69
-- opt.deltaL1 = 0.0001

-- optEnc.learningRate = 0.0002
-- optDec.learningRate = 0.0002
-- optAdvGen.learningRate = 0.0002
-- optAdv.learningRate = 0.0002
-- model_opts.styleRatio = 1E-4

-- criterion = criterion_label

-- opt.saveStateIter = 50
-- opt.nepochs = 500

-- opt.saveProgressIter = 1


-- dataProvider = DataProvider.create(model_opts.image_dir, model_opts)
-- shape_embeddings = torch.load(embedding_file)

-- function dataProvider:getCodes(indices, train_or_test)
--     local codes = shape_embeddings[train_or_test]:index(1, indices):cuda()
--     return codes
-- end

-- criterion = criterion_label

-- print(opt)

-- epoch_tmp = 500
-- if epoch_tmp >= opt.epoch then
--     print('Training conditional autoencoder')
--     opt.nepochs = opt.epoch + opt.saveStateIter
    
--     if opt.epoch ~= opt.nepochs then  


--         setup.getModel()
--         learner.loop()

--         -- clear all of the big variables from memory
--         encoder = nil  
--         decoder = nil  
--         adversary = nil
--         theta, gradTheta, thetaAdv, gradThetaAdv = nil, nil, nil, nil
--         os.exit()
--     end
-- end

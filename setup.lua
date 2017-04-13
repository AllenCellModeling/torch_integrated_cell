require 'optim'

setup = {} 

function setup.init(opts)
    
    opts.save = {}
    opts.save.opts = opts.saveDir .. '/opts.t7'

    opts.save.enc = opts.saveDir .. '/enc.t7'
    opts.save.dec = opts.saveDir .. '/dec.t7'
    opts.save.encD = opts.saveDir .. '/encD.t7'
    opts.save.decD = opts.saveDir .. '/decD.t7'

    opts.save.optEnc = opts.saveDir .. '/optEnc.t7'
    opts.save.optDec = opts.saveDir .. '/optDec.t7'
    opts.save.optEncD = opts.saveDir .. '/optEncD.t7'
    opts.save.optDecD = opts.saveDir .. '/optDecD.t7'

    opts.save.stateEnc = opts.saveDir .. '/stateEnc.t7'
    opts.save.stateDec = opts.saveDir .. '/stateDec.t7'
    opts.save.stateEncD = opts.saveDir .. '/stateEncD.t7'
    opts.save.stateDecD = opts.saveDir .. '/stateDecD.t7'

    opts.save.rng = opts.saveDir .. '/rng.t7'
    opts.save.rngCuda = opts.saveDir .. '/rng_cuda.t7'

    opts.save.tmpEmbeddings = opts.saveDir .. '/tmp_embeddings.t7'
    opts.save.tmpPlots = opts.saveDir .. '/tmp_plots.t7'
    opts.save.tmpEpoch = opts.saveDir .. '/tmp_epoch.t7'

    opts.save.plots = opts.saveDir .. '/plots.t7'
    opts.save.epoch = opts.saveDir .. '/epoch.t7'
    
    opts.save.data = opts.dataDir .. '/data_' .. opts.imsize .. '.t7'
    
    opts.save.loggerTrain = opts.saveDir .. '/logger_train.t7'
    opts.save.tmpLoggerTrain = opts.saveDir .. '/tmp_logger_train.t7'
    
    opts.save.loggerTest = opts.saveDir .. '/logger_test.t7'
    opts.save.tmpLoggerTest = opts.saveDir .. '/tmp_logger_test.t7'
    
    
    if opts.skipGanD then
        opts.useGanD = false
    else
        opts.useGanD = true
    end
    opts.image_sub_size = 1.4707*(512/opts.imsize)
    
        
    opts.epoch = 0

    optEnc = {}
    optEnc.optimizer = opts.optimizer
    optEnc.learningRate = opts.learningRate
    optEnc.beta1 = opts.beta1  

    optDec = {}
    optDec.optimizer = opts.optimizer
    optDec.learningRate = opts.learningRate
    optDec.beta1 = opts.beta1

    optAdv = {}
    optAdv.optimizer = opts.optimizer
    optAdv.learningRate = opts.learningRate
    optAdv.beta1 = opts.beta1 

    optAdvGen = {}
    optAdvGen.optimizer = opts.optimizer
    optAdvGen.learningRate = opts.learningRate
    optAdvGen.beta1 = opts.beta1    

    stateEnc = {}
    stateDec = {}
    stateAdv = {}
    stateAdvGen = {}

    paths.mkdir(opts.dataDir)
    
    loggerTrain = optim.Logger()
    loggerTrain = loggerTrain:setNames{'epoch', 'xHat loss', 'label loss', 'zHat loss', 'advEnc loss', 'advDec loss', 'minimaxEnc loss', 'minimaxDec loss', 'time'}
    
    loggerTest = optim.Logger()
    loggerTest = loggerTrain:setNames{'epoch', 'xHat loss', 'label loss', 'zHat loss'}
    
     if paths.filep(opts.save.opts) then
        print('Loading previous optimizer state')

        opts = torch.load(opts.save.opts)

        optEnc = torch.load(opts.save.optEnc)
        optDec = torch.load(opts.save.optDec)
        optAdv = torch.load(opts.save.optEncD)
        optAdvGen = torch.load(opts.save.optDecD)        

        stateEnc = utils.table2cuda(torch.load(opts.save.stateEnc))
        stateDec = utils.table2cuda(torch.load(opts.save.stateDec))    
        stateAdv = utils.table2cuda(torch.load(opts.save.stateEncD))
        
        -- plots = torch.load(opts.save.plots)
        
        loggerTest = torch.load(opts.save.loggerTest)
        loggerTest.file = io.stdout
        
        loggerTrain = torch.load(opts.save.loggerTrain)
        loggerTrain.file = io.stdout
        
        if paths.filep(opts.save.stateDecD) then
            stateAdvGen = utils.table2cuda(torch.load(opts.save.stateDecD))
        end
    end

    -- plots = {}
    -- loss, advloss, advlossGen = {}, {}, {}
    
    return opts
end

function setup.getModel(opts)

    print('Loading model type: ' .. opts.modelName)

    package.loaded['models/' .. opts.modelName] = nil
    Model = require ('models/' .. opts.modelName)
    autoencoder = nil
    encoder = nil
    decoder = nil

    if paths.filep(opts.save.dec) then
        print('Loading model from ' .. opts.saveDir)
        
        print('Loading encoder')
        encoder = torch.load(opts.save.enc)
        encoder:float()
        encoder:clearState()
        collectgarbage()
        
        print('Loading adversary')
        adversary = torch.load(opts.save.encD)
        adversary:float()
        adversary:clearState()
        collectgarbage()
        
        if paths.filep(opts.save.decD) then
            print('Loading adversaryGen')
            adversaryGen = torch.load(opts.save.decD)
            adversaryGen:float()
            adversaryGen:clearState()
            collectgarbage()
        else
            adversaryGen = nil
        end
        
        print('Loading decoder')
        decoder = torch.load(opts.save.dec)    
        decoder:float()
        decoder:clearState()
        collectgarbage()
        
        
        
        if paths.filep(opts.save.rng) then
            print('Loading RNG')
            torch.setRNGState(torch.load(opts.save.rng))
            cutorch.setRNGState(torch.load(opts.save.rngCuda))
        end
            
        
         -- print(gpuIDs)
        setGpuID = function(model, gpuIDs)
            for i = 1, #model.modules do
                model.modules[i].device = gpuIDs[i]
                
                if i == #model.modules then
                    model.modules[i].outdevice = opts.gpu1
                else
                    model.modules[i].outdevice = gpuIDs[i+1]
                end
            end
            return model
        end

        gpuIDs = torch.zeros(#encoder.modules):fill(opts.gpu1)
        -- gpuIDs:sub(9,#encoder.modules):fill(gpu2)
        encoder = setGpuID(encoder, gpuIDs)

        gpuIDs = torch.zeros(#decoder.modules):fill(opts.gpu2)
        decoder = setGpuID(decoder, gpuIDs)

        gpuIDs = torch.zeros(#adversary.modules):fill(opts.gpu2)
        adversary = setGpuID(adversary, gpuIDs)

        if adversaryGen ~= nil then
            gpuIDs = torch.zeros(#adversaryGen.modules):fill(opts.gpu3)
            adversaryGen = setGpuID(adversaryGen, gpuIDs)
        end
        
        print('Done loading model')
    else
        print('Creating new model')
        
        print(opts)
        Model:create(opts)
        paths.mkdir(opts.saveDir)
        print('Done creating model')
        
        decoder = Model.decoder
        encoder = Model.encoder
        adversary = Model.adversary
        adversaryGen = Model.adversaryGen
        

        -- print(gpuIDs)
        setGpuID = function(model, gpuIDs)
            for i = 1, #model.modules do
                model.modules[i] = nn.GPU(model.modules[i], gpuIDs[i])
                -- model.modules[i].device = gpuIDs[i]
                if i == #model.modules then
                    model.modules[i].outdevice = opts.gpu1
                else
                    model.modules[i].outdevice = gpuIDs[i+1]
                end
            end
            return model
        end

        gpuIDs = torch.zeros(#encoder.modules):fill(opts.gpu1)
        -- gpuIDs:sub(4,#encoder.modules):fill(gpu2)
        encoder = setGpuID(encoder, gpuIDs)

        gpuIDs = torch.zeros(#decoder.modules):fill(opts.gpu2)
        decoder = setGpuID(decoder, gpuIDs)

        gpuIDs = torch.zeros(#adversary.modules):fill(opts.gpu2)
        adversary = setGpuID(adversary, gpuIDs)

        gpuIDs = torch.zeros(#adversaryGen.modules):fill(opts.gpu3)
        adversaryGen = setGpuID(adversaryGen, gpuIDs)

        opts_optnet = {inplace=true, mode='training'}
        
        optnet.optimizeMemory(encoder, dataProvider:getImages(torch.LongTensor{1,2}, 'train'), opts_optnet)
        optnet.optimizeMemory(decoder, encoder.output, opts_optnet)
        optnet.optimizeMemory(adversary, encoder.output[#encoder.output], opts_optnet)
        optnet.optimizeMemory(adversaryGen, dataProvider:getImages(torch.LongTensor{1,2}, 'train'), opts_optnet)

        Model = nil
    end

    criterion_out = nn.BCECriterion()
    criterion_label = nn.ClassNLLCriterion()
    criterion_other = nn.MSECriterion()
    criterion_latent = nn.MSECriterion()

    criterion_adv = nn.BCECriterion()
    criterionAdvGen = nn.BCECriterion()

    if cuda then
        -- set which parts of the model are on which gpu
        print('Converting to cuda')
        
        encoder:cuda()
        decoder:cuda()
        adversary:cuda()
        
        if adversaryGen ~= nil then
            adversaryGen:cuda()
        end    
        
        criterion_label:cuda()
        criterion_other:cuda()
        criterion_latent:cuda()
        criterion_out:cuda()
        criterion_adv:cuda()
        criterionAdvGen:cuda()

        print('Done converting to cuda')
    end
    
    if opts.test_model then

        print('Data size')
        print(dataProvider.train.inds:size())

        encoder:evaluate()
        decoder:evaluate()        
        adversary:evaluate()
        
        local im_in, im_out = dataProvider:getImages(torch.LongTensor{1}, 'train')
        local label = dataProvider:getLabels(torch.LongTensor{1}, 'train'):clone()
        
        print('Testing encoder')
        local code = encoder:forward(im_in:cuda());
    
        print(im_in:type())
        print(im_in:size())

        print('Code size:')
        print(code)

        print(label)

        print(torch.cat(code, 2):size())

        print('Testing decoder')
        
        local im_out_hat = decoder:forward(code)

        print('Out size:')
        print(im_out_hat:size())

        print(criterion_out:forward(im_out_hat, im_out:cuda()))

        -- itorch.image(imtools.im2projection(im_out))
        -- itorch.image(imtools.im2projection(im_out_hat))

        print('Testing adversary')
        print(adversary:forward(code[#code]))

        encoder:training()
        decoder:training()
        adversary:training()
        adversaryGen:training()
    end    
    
    cudnn.benchmark = opts.cudnnBenchmark or false
    cudnn.fastest = opts.cudnnFastest or false
    
    cudnn.convert(encoder, cudnn)
    cudnn.convert(decoder, cudnn)
    cudnn.convert(adversary, cudnn)
    
    if adversaryGen ~= nil then
        cudnn.convert(adversaryGen, cudnn)
    end

    print('Done getting parameters')
end


return setup



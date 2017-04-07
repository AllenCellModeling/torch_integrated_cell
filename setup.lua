setup = {}


function setup.getModelOpts()

    local model_opts = {}
    model_opts.cuda = cuda
    
    model_opts.parent_dir = 'AAE_shape_learner_v2'
    model_opts.model_name = 'cell_learner_3_caae_learner'
    model_opts.save_dir = model_opts.parent_dir .. '/' .. model_opts.model_name
    model_opts.image_dir = '/root/images/2016_11_08_Nuc_Cell_Seg_8_cell_lines_V22/processed_aligned/2D'
    model_opts.image_sub_size = 16.96875/2

    model_opts.nLatentDims = 16
    model_opts.channel_inds_in = torch.LongTensor{1,3}
    model_opts.channel_inds_out = torch.LongTensor{1,3}

    model_opts.rotate = true
    model_opts.nChIn = model_opts.channel_inds_in:size(1)
    model_opts.nChOut = model_opts.channel_inds_out:size(1)
    model_opts.nClasses = 0
    model_opts.nOther = 0
    model_opts.dropoutRate = 0.2
    model_opts.fullConv = true
    
    model_opts.adversarialGen = false
    
    model_opts.test_model = false
    model_opts.verbose = true

    paths.mkdir(model_opts.parent_dir)

    return model_opts
end



function setup.getModel()

    print('Loading model type: ' .. model_opts.model_name)

    package.loaded['models/' .. model_opts.model_name] = nil
    Model = require ('models/' .. model_opts.model_name)
    autoencoder = nil
    encoder = nil
    decoder = nil

    if paths.filep(model_opts.save_dir .. '/decoder.t7') then
        print('Loading model from ' .. model_opts.save_dir)
        
        print('Loading encoder')
        encoder = torch.load(model_opts.save_dir .. '/encoder.t7')
        encoder:float()
        encoder:clearState()
        collectgarbage()
        
        print('Loading adversary')
        adversary = torch.load(model_opts.save_dir .. '/adversary.t7')
        adversary:float()
        adversary:clearState()
        collectgarbage()
        
        if paths.filep(model_opts.save_dir .. '/adversaryGen.t7') then
            print('Loading adversaryGen')
            adversaryGen = torch.load(model_opts.save_dir .. '/adversaryGen.t7')
            adversaryGen:float()
            adversaryGen:clearState()
            collectgarbage()
        else
            adversaryGen = nil
        end
        
        print('Loading decoder')
        decoder = torch.load(model_opts.save_dir .. '/decoder.t7')    
        decoder:float()
        decoder:clearState()
        collectgarbage()
        
        
        
        if paths.filep(model_opts.save_dir .. '/rng.t7') then
            print('Loading RNG')
            torch.setRNGState(torch.load(model_opts.save_dir .. '/rng.t7'))
            cutorch.setRNGState(torch.load(model_opts.save_dir .. '/rng_cuda.t7'))
        end
            
        
         -- print(gpuIDs)
        set_gpu_id = function(model, gpuIDs)
            for i = 1, #model.modules do
                model.modules[i].device = gpuIDs[i]
                
                if i == #model.modules then
                    model.modules[i].outdevice = gpu1
                else
                    model.modules[i].outdevice = gpuIDs[i+1]
                end
            end
            return model
        end

        gpuIDs = torch.zeros(#encoder.modules):fill(gpu1)
        -- gpuIDs:sub(9,#encoder.modules):fill(gpu2)
        encoder = set_gpu_id(encoder, gpuIDs)

        gpuIDs = torch.zeros(#decoder.modules):fill(gpu2)
        decoder = set_gpu_id(decoder, gpuIDs)

        gpuIDs = torch.zeros(#adversary.modules):fill(gpu2)
        adversary = set_gpu_id(adversary, gpuIDs)

        if adversaryGen ~= nil then
            gpuIDs = torch.zeros(#adversaryGen.modules):fill(gpu3)
            adversaryGen = set_gpu_id(adversaryGen, gpuIDs)
        end
        
        print('Done loading model')
    else
        print('Creating new model')
        print(model_opts)
        Model:create(model_opts)
        paths.mkdir(model_opts.save_dir)
        print('Done creating model')
        
        decoder = Model.decoder
        encoder = Model.encoder
        adversary = Model.adversary
        adversaryGen = Model.adversaryGen
        

        -- print(gpuIDs)
        set_gpu_id = function(model, gpuIDs)
            for i = 1, #model.modules do
                model.modules[i] = nn.GPU(model.modules[i], gpuIDs[i])
                -- model.modules[i].device = gpuIDs[i]
                if i == #model.modules then
                    model.modules[i].outdevice = gpu1
                else
                    model.modules[i].outdevice = gpuIDs[i+1]
                end
            end
            return model
        end

        gpuIDs = torch.zeros(#encoder.modules):fill(gpu1)
        -- gpuIDs:sub(4,#encoder.modules):fill(gpu2)
        encoder = set_gpu_id(encoder, gpuIDs)

        gpuIDs = torch.zeros(#decoder.modules):fill(gpu2)
        decoder = set_gpu_id(decoder, gpuIDs)

        gpuIDs = torch.zeros(#adversary.modules):fill(gpu2)
        adversary = set_gpu_id(adversary, gpuIDs)

        gpuIDs = torch.zeros(#adversaryGen.modules):fill(gpu3)
        adversaryGen = set_gpu_id(adversaryGen, gpuIDs)

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
    
    if model_opts.test_model then

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
    
    cudnn.benchmark = false
    cudnn.fastest = false
    
    cudnn.convert(encoder, cudnn)
    cudnn.convert(decoder, cudnn)
    cudnn.convert(adversary, cudnn)
    
    if adversaryGen ~= nil then
        cudnn.convert(adversaryGen, cudnn)
    end

    print('Done getting parameters')
end

function setup.getLearnOpts(model_opts)

    opt_path = model_opts.save_dir .. '/opt.t7'
    optEnc_path = model_opts.save_dir .. '/optEnc.t7'
    optDec_path = model_opts.save_dir .. '/optDec.t7'
    optAdv_path = model_opts.save_dir .. '/optD.t7'
    optAdvGen_path = model_opts.save_dir .. '/optAdvGen.t7'
    
    stateEnc_path = model_opts.save_dir .. '/stateEnc.t7'
    stateDec_path = model_opts.save_dir .. '/stateDec.t7'
    stateAdv_path = model_opts.save_dir .. '/stateD.t7'
    stateAdvGen_path = model_opts.save_dir .. '/stateAdvGen.t7'

    plots_path = model_opts.save_dir .. '/plots.t7'
    
    if paths.filep(opt_path) then
        print('Loading previous optimizer state')

        opt = torch.load(opt_path)

        optEnc = torch.load(optEnc_path)
        optDec = torch.load(optDec_path)
        optAdv = torch.load(optAdv_path)
        optAdvGen = torch.load(optAdv_path)        

        
        stateEnc = utils.table2cuda(torch.load(stateEnc_path))
        stateDec = utils.table2cuda(torch.load(stateDec_path))    
        stateAdv = utils.table2cuda(torch.load(stateAdv_path))
        
        
        plots = torch.load(plots_path)
        
        if paths.filep(stateAdvGen_path) then
            stateAdvGen = utils.table2cuda(torch.load(stateAdvGen_path))
            
            losses = plots[1][3]:totable()
            latentlosses = plots[2][3]:totable()
            advlosses = plots[3][3]:totable()
            advGenLosses = plots[4][3]:totable()
            advMinimaxLoss = plots[5][3]:totable()
            advGenMinimaxLoss = plots[6][3]:totable()
            reencodelosses = plots[7][3]:totable()
            
        else
            print('Could not find stateAdvGen_path: ' .. stateAdvGen_path)
            stateAdvGen = {}
            
            losses = plots[1][3]:totable()
            latentlosses = plots[2][3]:totable()
            advlosses = plots[3][3]:totable()
            advMinimaxLoss = plots[4][3]:totable()
            reencodelosses = plots[5][3]:totable()
        end

    else
        opt = {}

        opt.epoch = 0
        opt.nepochs = 2000
        opt.adversarial = true
        opt.adversarialGen = model_opts.adversarialGen

        opt.learningRateA = 0.01
        opt.learningRateAdv = 0.01

        opt.min_rateA =  0.0000001
        opt.min_rateD =  0.0001
        opt.learningRateDecay = 0.999
        opt.optimizer = 'adam'
        opt.batchSize = 64
        opt.verbose = model_opts.verbose
--         opt.updateD = 0.50
--         opt.updateA = 0.40
        opt.update_thresh = 0.58
        opt.saveProgressIter = 5
        opt.saveStateIter = 50


        optEnc = {}
        optEnc.optimizer = opt.optimizer
        optEnc.learningRate = opt.learningRateA
        optEnc.min_rateA =  0.0000001
        optEnc.beta1 = 0.5        
        -- optEnc.momentum = 0
        -- optEnc.numUpdates = 0
        -- optEnc.coefL2 = 0
        -- optEnc.coefL1 = 0

        optDec = {}
        optDec.optimizer = opt.optimizer
        optDec.learningRate = opt.learningRateA
        optDec.min_rateA =  0.0000001
        optDec.beta1 = 0.5
        -- optDec.momentum = 0
        -- optDec.numUpdates = 0
        -- optDec.coefL2 = 0
        -- optDec.coefL1 = 0
        
        optAdv = {}
        optAdv.optimizer = opt.optimizer
        optAdv.learningRate = opt.learningRateAdv
        optAdv.min_rateA =  0.0000001
        optAdv.beta1 = 0.5        
        -- optAdv.momentum = 0
        -- optAdv.numUpdates = 0
        -- optAdv.coefL2 = 0
        -- optAdv.coefL1 = 0
        
        optAdvGen = {}
        optAdvGen.optimizer = opt.optimizer
        optAdvGen.learningRate = opt.learningRateAdv
        optAdvGen.min_rateA =  0.0000001
        optAdvGen.beta1 = 0.5        
        -- optAdvGen.momentum = 0
        -- optAdvGen.numUpdates = 0
        -- optAdvGen.coefL2 = 0
        -- optAdvGen.coefL1 = 0

        stateEnc = {}
        stateDec = {}
        stateAdv = {}
        stateAdvGen = {}
        
        losses, latentlosses, advlosses, advGenLosses, advMinimaxLoss, advGenMinimaxLoss, advlossesGen, reencodelosses = {}, {}, {}, {}, {}, {}, {}, {}, {}
    end

    plots = {}
    loss, advloss, advlossGen = {}, {}, {}
end

return setup


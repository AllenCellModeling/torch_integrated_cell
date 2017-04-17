local learner = {}

function optim_step(net, loss, optParam, optStates)
-- this function assumes that all modules are nn.GPU-decorated
    local function feval_dummy(param)
        if thisparam ~= param then
            thisparam:copy(param)
        end
        return loss, thisgrad
    end

    local c = 1
    for i = 1, #net.modules do
        local gpu = net.modules[i].device
        cutorch.setDevice(gpu)

        local theta, gradTheta = net.modules[i]:parameters()

        for j = 1,#theta do
            thisparam = theta[j]
            thisgrad = gradTheta[j]

            optState = optStates[c] or {}

            for k, v in pairs(optState) do
                if type(v) == 'userdata' and v:getDevice() ~= gpu then
                    optState[k] = v:clone()
                end
            end

            optim[optParam.optimizer](feval_dummy, thisparam, optParam, optState) 
            optStates[c] = optState
            c = c+1
        end

        cutorch.setDevice(opts.gpu1)
    end
end

function learner.loop(nIn)
    local x -- Minibatch
    local label

    local xHatLoss = 0
    local labelLoss = 0
    local zHatLoss = 0
    
    local advEncLoss = 0
    local advDecLoss = 0
    
    local minimaxEncLoss = 0
    local minimaxDecLoss = 0    

    -- Create optimiser function evaluation
    local dAdvDz = function(params)
        adversary:zeroGradParameters()

        local input = torch.Tensor(opts.batchSize, opts.nLatentDims):normal(0, 1):typeAs(x_in)
        local label = torch.ones(opts.batchSize):typeAs(x_in) -- Labels for real samples

        local output = adversary:forward(input)
        local errEnc_real = criterion_adv:forward(output, label)
        local df_do = criterion_adv:backward(output, label)
        adversary:backward(input, df_do)

        codes = encoder:forward(x_in)
        local input = codes[#codes]
        label = torch.zeros(opts.batchSize):typeAs(x_in) -- Labels for generated samples

        output = adversary:forward(input)
        local errEnc_fake = criterion_adv:forward(output, label)
        local df_do = criterion_adv:backward(output, label)
        adversary:backward(input, df_do)

        advEncLoss = (errEnc_real + errEnc_fake)/2

        return advEncLoss, gradParametersAdv
    end
        
    local dAdvGenDx = function(params)   
        adversaryGen:zeroGradParameters()

        input = x_out
        if opts.nClasses > 0 then
            label = classLabel
        else
            label = torch.ones(opts.batchSize):typeAs(x_in) 
        end
        
        local output = adversaryGen:forward(input)
        local errD_real = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        adversaryGen:backward(input, df_do)

        zFake = {}
        c = 1
        if opts.nClasses > 0 then
            zFake[c] = classLabelOneHot
            c = c+1
        end
        if opts.nOther > 0 then
            zFake[c] = code
            c = c+1
        end
        
        zFake[c] = torch.Tensor(opts.batchSize, opts.nLatentDims):normal(0, 1):typeAs(x_in)

        input = decoder:forward(zFake)
        
        if opts.nClasses > 0 then
            label = torch.Tensor(opts.batchSize):typeAs(x_in):fill(opts.nClasses+1)
        else
            label = torch.zeros(opts.batchSize):typeAs(x_in) 
        end

        local output = adversaryGen:forward(input)
        local errD_fake = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        adversaryGen:backward(input, df_do)

        advDecLoss = (errD_real + errD_fake)/2

        return advDecLoss, gradParametersAdvGen
    end
    
    local dAutoencoderDx = function(params)
        encoder:zeroGradParameters()
        
        -- the encoder has already gone forward
        xHat = decoder:forward(codes)
        xHatLoss = criterion_out:forward(xHat, x_out)
        loss = xHatLoss
        local gradLoss = criterion_out:backward(xHat, x_out)

        -- Backwards pass
        encoder:backward(x_in, decoder:backward(codes, gradLoss))

        -- Now the regularization pass
        encoder_out = encoder.output

        c = 1
        gradLosses = {{}}  
        
        labelLoss = 0
        shapeLoss = 0

        if opts.nClasses > 0 then
            local labelHat = encoder_out[c]

            labelLoss = criterion_label:forward(labelHat, classLabel)
            local labelGradLoss = criterion_label:backward(labelHat, classLabel)

            -- loss = loss + labelLoss
            gradLosses[c] = labelGradLoss

            c = c+1
        end

        if opts.nOther > 0 then
            local zHat = encoder_out[c]
            zHatLoss = criterion_other:forward(zHat, code)
            local zHatGradLoss = criterion_other:backward(zHat, code)

            -- loss = loss + shapeLoss
            gradLosses[c] = zHatGradLoss

            c = c+1
        end

        local yReal = torch.ones(opts.batchSize):typeAs(x_in)
        -- Train autoencoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
        local predFake = adversary:forward(encoder.output[c])
        minimaxEncLoss = criterion_adv:forward(predFake, yReal)
        local gradMinimaxLoss = criterion_adv:backward(predFake, yReal)
        local gradMinimax = adversary:updateGradInput(encoder.output[c], gradMinimaxLoss) -- Do not calculate gradient wrt adversary parameters
        gradLosses[c] = gradMinimax*opts.advLatentRatio

        encoder:backward(x_in, gradLosses)

        latentLoss = minimaxEncLoss + zHatLoss + labelLoss
        
        loss = xHatLoss + latentLoss
    --     if fakeLoss < 0.79 and realLoss < 0.79 then -- -ln(0.45)
        cutorch.synchronizeAll()
        return loss, gradThetaEnc
    end
        
    local dDecdAdvGen = function(params)        
        decoder:zeroGradParameters()

        label = nil
        if opts.nClasses > 0 then
            label = classLabel
        else
            label = torch.ones(opts.batchSize):typeAs(x_in) 
        end

        local output = adversaryGen.output -- netD:forward(input) was already executed in fDx, so save computation
        minimaxDecLoss = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        local df_dg = adversaryGen:updateGradInput(input, df_do):clone()

        adversaryGen:clearState()

        decoder:backward(zFake, df_dg*opts.advGenRatio)
        return minimaxDecLoss, gradParametersG
    end
    
    local dDecdAdvGen2 = function(params)        
        if opts.nClasses > 0 then
            label = classLabel
        else
            label = torch.ones(opts.batchSize):typeAs(x_in) 
        end

        -- local xHat = decoder:forward(codes)
        local output = adversaryGen:forward(xHat) -- netD:forward(input) was already executed in fDx, so save computation
        minimaxDecLoss2 = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_dg = adversaryGen:updateGradInput(xHat, df_do)

        decoder:backward(codes, df_dg*opts.advGenRatio)
        return minimaxDecLoss, gradParametersG
    end    
    
    local ndat = nIn or dataProvider.train.inds:size()[1]
    -- main loop
    print("Starting learning")
    while opts.epoch < opts.nepochs do
        local tic = torch.tic()
        
        opts.epoch = opts.epoch+1

        local indices = torch.randperm(ndat):long():split(opts.batchSize)
        indices[#indices] = nil
        local N = #indices * opts.batchSize
        
        -- variable to hold embeddings every loop
        embeddings = {}
        embeddings['train'] = torch.zeros(ndat, opts.nLatentDims)
        
        xHatLosses = {}
        labelLosses = {}
        zHatLosses = {}
        advEncLosses = {}
        advDecLosses = {}
        minimaxEncLosses = {}
        minimaxDecLosses = {}

        local start = 1

        for t,v in ipairs(indices) do
            collectgarbage()
            
            local stop = start + v:size(1) - 1

            x_in, x_out = dataProvider:getImages(v, 'train')
            -- Forward pass
            x_in = x_in:cuda()
            x_out = x_out:cuda()
            
            if opts.nOther > 0 then
                classLabelOneHot = dataProvider:getLabels(v, 'train'):cuda()
                __, classLabel = torch.max(classLabelOneHot, 2)
                classLabel = torch.squeeze(classLabel:typeAs(x_in))
                
                classLabelOneHot = torch.log(classLabelOneHot)
                classLabelOneHot:maskedFill(classLabelOneHot:eq(-math.huge), -25)
            end

            if opts.nClasses > 0 then
                code = dataProvider:getCodes(v, 'train')
            end

            -- update the decoder's advarsary            
            if opts.useGanD then
                dAdvGenDx()            
                optim_step(adversaryGen, advDecLoss, optAdvGen, stateAdvGen)
            end
            
            -- update the encoder's advarsary
            dAdvDz()
            optim_step(adversary, advEncLoss, optAdv, stateAdv)
            adversary:clearState()
            
            
            if opts.useGanD then
                dDecdAdvGen()
            end
            dAutoencoderDx()

            if opts.useGanD then
                dDecdAdvGen2()
            end
            
            minimaxDecLoss = minimaxDecLoss+minimaxDecLoss2;
            
            optim_step(encoder, loss, optEnc, stateEnc)
            optim_step(decoder, xHatLoss+minimaxDecLoss, optDec, stateDec)
            
            xHatLosses[#xHatLosses+1] = xHatLoss
            labelLosses[#labelLosses+1] = labelLoss
            zHatLosses[#zHatLosses+1] = zHatLoss
            advEncLosses[#advEncLosses+1] = advEncLoss
            advDecLosses[#advDecLosses+1] = advDecLoss
            minimaxEncLosses[#minimaxEncLosses+1] = minimaxEncLoss
            minimaxDecLosses[#minimaxDecLosses+1] = minimaxDecLoss

            embeddings['train']:indexCopy(1, v, torch.Tensor(codes[#codes]:size()):copy(codes[#codes]))
            
            loggerTrain:add{opts.epoch, xHatLoss, labelLoss, zHatLoss, advEncLoss, advDecLoss, minimaxEncLoss, minimaxDecLoss, torch.toc(tic)}
            
            encoder:clearState()
            decoder:clearState()
            adversaryGen:clearState()
            adversary:clearState()
        end
        collectgarbage()

        x_in, x_out = nil, nil
        
        m_xHat =     torch.mean(torch.Tensor(xHatLosses))
        m_label =    torch.mean(torch.Tensor(labelLosses))
        m_zHat =     torch.mean(torch.Tensor(zHatLosses))
        m_advEnc =   torch.mean(torch.Tensor(advEncLosses))
        m_advDec =   torch.mean(torch.Tensor(advDecLosses))
        m_mmEnc =    torch.mean(torch.Tensor(minimaxEncLosses))
        m_mmDec =    torch.mean(torch.Tensor(minimaxDecLosses))
        
        if m_xHat == math.huge or m_xHat ~= m_xHat or advEncLoss == math.huge or advEncLoss ~= advEncLoss then
            print('Exiting')
            break
        end

        if opts.epoch % opts.saveProgressIter == 0 then
            plotStuff()
        end
            
        if opts.epoch % opts.saveStateIter == 0 then
            saveStuff()
        end
        
        plots = nil
    end
end

function plotStuff()
    encoder:evaluate()
    decoder:evaluate()
    rotate_tmp = opts.rotate
    dataProvider.opts.rotate = false

    local x_in, x_out = dataProvider:getImages(torch.linspace(1,10,10):long(), 'train')
    recon_train = evalIm(x_in,x_out, opts)

    local x_in, x_out = dataProvider:getImages(torch.linspace(1,10,10):long(), 'test')
    recon_test = evalIm(x_in,x_out, opts)

    local reconstructions = torch.cat(recon_train, recon_test,2)
    image.save(opts.saveDir .. '/progress_' .. opts.epoch .. '.png', reconstructions)

    -- traintest = {'test'}
    -- for i = 1,#traintest do
        -- train_or_test = traintest[i]
    train_or_test = 'test'

    local ndat = dataProvider[train_or_test].inds:size()[1]

    embeddings[train_or_test] = torch.zeros(ndat, opts.nLatentDims)
    local indices = torch.linspace(1,ndat,ndat):long():split(opts.batchSize)
    -- print('indices: ' .. #indices)
    
    labelLoss = torch.zeros(#indices)
    zHatLoss = torch.zeros(#indices)
    xHatLoss = torch.zeros(#indices)
    
    
    local start = 1
    local c = 1
    for t,v in ipairs(indices) do
        collectgarbage()
        local stop = start + v:size(1) - 1

        local x_in, x_out = dataProvider:getImages(v, train_or_test)
        local x_in = x_in:cuda()
        local x_out = x_out:cuda()

        local encoder_out = encoder:forward(x_in)
        local xHat = decoder:forward(encoder_out)
        iter = 1;
        
        if opts.nClasses > 0 then
            local classLabelOneHot = dataProvider:getLabels(v, 'train'):cuda()
            local __, classLabel = torch.max(classLabelOneHot, 2)
            classLabel = torch.squeeze(classLabel:typeAs(x_in))

            local labelHat = encoder_out[iter]
            labelLoss[c] = criterion_label:forward(labelHat, classLabel)

            iter = iter+1
        end

        if opts.nOther > 0 then
            local code = dataProvider:getCodes(v, train_or_test)
            local zHat = encoder_out[iter]
            zHatLoss[c] = criterion_other:forward(zHat, code)

            iter = iter+1
        end


        xHatLoss[c] = criterion_out:forward(xHat, x_out)        
        embeddings['test']:indexCopy(1, v, torch.Tensor(encoder_out[iter]:size()):copy(encoder_out[iter]))

        start = stop + 1
        c = c+1
    end

    loggerTest:add{opts.epoch, torch.mean(xHatLoss), torch.mean(labelLoss), torch.mean(zHatLoss)}

    dataProvider.opts.rotate = rotate_tmp

    encoder:training()
    decoder:training()          

    torch.save(opts.save.tmpLoggerTest, loggerTest)
    torch.save(opts.save.tmpLoggerTrain, loggerTrain)
    torch.save(opts.save.tmpEpoch, opts.epoch)
    
    torch.save(opts.save.tmpEmbeddings, embeddings)
    embeddings = nil
end

function saveStuff()
    print('Saving model.')

    -- save the optimizer states
    torch.save(opts.save.stateEnc, utils.table2float(stateEnc))
    torch.save(opts.save.stateDec, utils.table2float(stateDec))            
    torch.save(opts.save.stateEncD, utils.table2float(stateAdv))
    torch.save(opts.save.stateDecD, utils.table2float(stateAdvGen))

    -- save the options
    torch.save(opts.save.opts, opts)
    torch.save(opts.save.optEnc, optEnc)
    torch.save(opts.save.optDec, optDec)
    torch.save(opts.save.optEncD, optAdv)
    torch.save(opts.save.optDecD, optAdvGen)  

    decoder:clearState()
    encoder:clearState()
    adversary:clearState()
    adversaryGen:clearState()            

    -- torch.save(opts.save.plots, plots)
    torch.save(opts.save.loggerTest, loggerTest)
    torch.save(opts.save.loggerTrain, loggerTrain)    
    torch.save(opts.save.epoch, opts.epoch)

    torch.save(opts.save.enc, encoder:float())            
    torch.save(opts.save.dec, decoder:float())
    torch.save(opts.save.encD, adversary:float())
    torch.save(opts.save.decD, adversaryGen:float())

    torch.save(opts.save.rng, torch.getRNGState())
    torch.save(opts.save.rngCuda, cutorch.getRNGState())

    decoder:cuda()
    encoder:cuda()
    adversary:cuda()
    adversaryGen:cuda()
end

return learner
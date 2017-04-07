-- require 'optim2'
require 'optnet'

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

        cutorch.setDevice(gpu1)
    end
end

function updateL1(net, deltaL1)
    if net.modules ~= nil then
        
        for i, module in ipairs(net.modules) do
            updateL1(module, deltaL1)
        end
    elseif torch.type(net):find('L1Penalty') then
        newL1 = net.l1weight + deltaL1
        if newL1 < 0 then
            newL1 = 0
        end
        net.l1weight = newL1
    end
end

function learner.loop(nIn)
    local x -- Minibatch
    local label

    local advLoss = 0
    local advGenLoss = 0
    local minimaxLoss = 0
    local reconLoss = 0
    local latentLoss = 0
    local reencodeLoss = 0
    local minimaxDecLoss = 0
    local minimaxDecLoss2 = 0
    -- Create optimiser function evaluation

    local dAdvDz = function(params)
        adversary:zeroGradParameters()

        local input = torch.Tensor(opt.batchSize, model_opts.nLatentDims):normal(0, 1):typeAs(x_in)
        local label = torch.ones(opt.batchSize):typeAs(x_in) -- Labels for real samples

        local output = adversary:forward(input)
        local errEnc_real = criterion_adv:forward(output, label)
        local df_do = criterion_adv:backward(output, label)
        adversary:backward(input, df_do)

        codes = encoder:forward(x_in)
        local input = codes[#codes]
         label = torch.zeros(opt.batchSize):typeAs(x_in) -- Labels for generated samples

         output = adversary:forward(input)
        local errEnc_fake = criterion_adv:forward(output, label)
        local df_do = criterion_adv:backward(output, label)
        adversary:backward(input, df_do)

        advLoss = (errEnc_real + errEnc_fake)/2

        return advLoss, gradParametersAdv
    end
        
    local dAdvGenDx = function(params)   
        adversaryGen:zeroGradParameters()

        input = x_out
        if model_opts.nClasses > 0 then
            label = classLabel
        else
            label = torch.ones(opt.batchSize):typeAs(x_in) 
        end
        
        local output = adversaryGen:forward(input)
        local errD_real = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        adversaryGen:backward(input, df_do)

        zFake = {}
        c = 1
        if model_opts.nClasses > 0 then
            zFake[c] = classLabelOneHot
            c = c+1
        end
        if model_opts.nOther > 0 then
            zFake[c] = code
            c = c+1
        end
        
        zFake[c] = torch.Tensor(opt.batchSize, model_opts.nLatentDims):normal(0, 1):typeAs(x_in)

        input = decoder:forward(zFake)
        
        if model_opts.nClasses > 0 then
            label = torch.Tensor(opt.batchSize):typeAs(x_in):fill(model_opts.nClasses+1)
        else
            label = torch.zeros(opt.batchSize):typeAs(x_in) 
        end

        local output = adversaryGen:forward(input)
        local errD_fake = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        adversaryGen:backward(input, df_do)

        advGenLoss = (errD_real + errD_fake)/2

        return advGenLoss, gradParametersAdvGen
    end
    
    local dAutoencoderDx = function(params)
        -- if thetaEnc ~= params then
        --     thetaEnc:copy(params)
        -- end

        encoder:zeroGradParameters()
        -- decoder:zeroGradParameters()
        -- adversary:zeroGradParameters()
        
        -- the encoder has already gone forward
        xHat = decoder:forward(codes)

        -- Versus criterion
        -- x_out = nn.ReLU(true):cuda():forward(x_out)
        
        reconLoss = criterion_out:forward(xHat, x_out)
        loss = reconLoss
        local gradLoss = criterion_out:backward(xHat, x_out)

        -- Backwards pass
        encoder:backward(x_in, decoder:backward(codes, gradLoss))

        -- Now the regularization pass
        encoder_out = encoder.output

        c = 1
        gradLosses = {{}}  
        
        labelLoss = 0
        shapeLoss = 0

        if model_opts.nClasses > 0 then
            local labelHat = encoder_out[c]

            labelLoss = criterion_label:forward(labelHat, classLabel)
            local labelGradLoss = criterion_label:backward(labelHat, classLabel)

            -- loss = loss + labelLoss
            gradLosses[c] = labelGradLoss

            c = c+1
        end

        if model_opts.nOther > 0 then
            local shapeHat = encoder_out[c]
            shapeLoss = criterion_other:forward(shapeHat, code)
            local shapeGradLoss = criterion_other:backward(shapeHat, code)

            -- loss = loss + shapeLoss
            gradLosses[c] = shapeGradLoss

            c = c+1
        end

        local yReal = torch.ones(opt.batchSize):typeAs(x_in)
        -- Train autoencoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
        local predFake = adversary:forward(encoder.output[c])
        minimaxLoss = criterion_adv:forward(predFake, yReal)
        local gradMinimaxLoss = criterion_adv:backward(predFake, yReal)
        local gradMinimax = adversary:updateGradInput(encoder.output[c], gradMinimaxLoss) -- Do not calculate gradient wrt adversary parameters
        gradLosses[c] = gradMinimax*model_opts.advLatentRatio

        -- gradLosses[c] = nn.Clip(-1, 1):cuda():forward(gradLosses[c])
        
        encoder:backward(x_in, gradLosses)

        latentLoss = minimaxLoss + shapeLoss + labelLoss
        
        loss = reconLoss + latentLoss
    --     if fakeLoss < 0.79 and realLoss < 0.79 then -- -ln(0.45)
        cutorch.synchronizeAll()

        return loss, gradThetaEnc
    end
        
    local dDecdAdvGen = function(params)        
       decoder:zeroGradParameters()

       --[[ the three lines below were already executed in fDx, so save computation
       noise:uniform(-1, 1) -- regenerate random noise
       local fake = netG:forward(noise)
       input:copy(fake) ]]--
        label = nil
        if model_opts.nClasses > 0 then
            label = classLabel
        else
            label = torch.ones(opt.batchSize):typeAs(x_in) 
        end
        
       local output = adversaryGen.output -- netD:forward(input) was already executed in fDx, so save computation
       minimaxDecLoss = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       local df_dg = adversaryGen:updateGradInput(input, df_do):clone()

        adversaryGen:clearState()
        
       decoder:backward(zFake, df_dg*model_opts.advGenRatio)
       return minimaxDecLoss, gradParametersG
        
    end
    
    local dDecdAdvGen2 = function(params)        

       --[[ the three lines below were already executed in fDx, so save computation
       noise:uniform(-1, 1) -- regenerate random noise
       local fake = netG:forward(noise)
       input:copy(fake) ]]--
        if model_opts.nClasses > 0 then
            label = classLabel
        else
            label = torch.ones(opt.batchSize):typeAs(x_in) 
        end
        
        -- local xHat = decoder:forward(codes)
        
       local output = adversaryGen:forward(xHat) -- netD:forward(input) was already executed in fDx, so save computation
       minimaxDecLoss2 = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       df_dg = adversaryGen:updateGradInput(xHat, df_do)

       decoder:backward(codes, df_dg*model_opts.advGenRatio)
       return minimaxDecLoss, gradParametersG
        
    end    
    
    
    local ndat = nIn or dataProvider.train.inds:size()[1]
    -- main loop
    print("Starting learning")
    while opt.epoch < opt.nepochs do
        local tic = torch.tic()
        
        opt.epoch = opt.epoch+1

        local indices = torch.randperm(ndat):long():split(opt.batchSize)
        indices[#indices] = nil
        local N = #indices * opt.batchSize

        for t,v in ipairs(indices) do
            collectgarbage()

            x_in, x_out = dataProvider:getImages(v, 'train')
            -- Forward pass
            x_in = x_in:cuda()
            x_out = x_out:cuda()
            
            
            if model_opts.nOther > 0 then
                classLabelOneHot = dataProvider:getLabels(v, 'train'):cuda()
                __, classLabel = torch.max(classLabelOneHot, 2)
                classLabel = torch.squeeze(classLabel:typeAs(x_in))
                
                classLabelOneHot = torch.log(classLabelOneHot)
                classLabelOneHot:maskedFill(classLabelOneHot:eq(-math.huge), -25)

            end

            if model_opts.nClasses > 0 then
                code = dataProvider:getCodes(v, 'train')
            end

            -- decoder:zeroGradParameters()
            -- update the decoder's advarsary
            
            if model_opts.useGanD then
                dAdvGenDx()            
                optim_step(adversaryGen, advGenLoss, optAdvGen, stateAdvGen)
            end
            
            -- update the encoder's advarsary
            dAdvDz()
            optim_step(adversary, advLoss, optAdv, stateAdv)
            adversary:clearState()
            
            if model_opts.useGanD then
                dDecdAdvGen()
            end
            dAutoencoderDx()
            -- codes = utils.shallowcopy(codes)
            -- encoder:clearState()
            
            if model_opts.useGanD then
                dDecdAdvGen2()
            end
            
            optim_step(encoder, loss, optEnc, stateEnc)
            optim_step(decoder, reconLoss+minimaxDecLoss+minimaxDecLoss2, optDec, stateDec)
            


            losses[#losses + 1] = reconLoss
            latentlosses[#latentlosses+1] = latentLoss
            reencodelosses[#reencodelosses+1] = reencodeLoss
            advlosses[#advlosses + 1] = advLoss    
            advGenLosses[#advGenLosses + 1] = advGenLoss
            
            advMinimaxLoss[#advMinimaxLoss + 1] = minimaxLoss
            advGenMinimaxLoss[#advGenMinimaxLoss + 1] = minimaxDecLoss+minimaxDecLoss2
            
                        
            encoder:clearState()
            decoder:clearState()
            adversaryGen:clearState()
            adversary:clearState()
        end
        collectgarbage()

        x_in, x_out = nil, nil

        recon_loss = torch.mean(torch.Tensor(losses)[{{-#indices,-1}}]);
        latent_loss = torch.mean(torch.Tensor(latentlosses)[{{#indices, -1}}])
        reencode_loss = torch.mean(torch.Tensor(reencodelosses)[{{#indices, -1}}])

        adv_loss = torch.mean(torch.Tensor(advlosses)[{{-#indices,-1}}]);
        advGen_loss = torch.mean(torch.Tensor(advGenLosses)[{{-#indices,-1}}]);
        minimax_latent_loss = torch.mean(torch.Tensor(advMinimaxLoss)[{{-#indices,-1}}]);
        minimax_gen_loss = torch.mean(torch.Tensor(advGenMinimaxLoss)[{{-#indices,-1}}]);

        print('Epoch ' .. opt.epoch .. '/' .. opt.nepochs .. ' Recon loss: ' .. recon_loss .. ' Adv loss: ' .. adv_loss .. ' AdvGen loss: ' .. advGen_loss .. ' time: ' .. torch.toc(tic))
        print(minimax_latent_loss)
        print(minimax_gen_loss)

        if recon_loss == math.huge or recon_loss ~= recon_loss or latent_loss == math.huge or latent_loss ~= latent_loss then
            print('Exiting')
            break
        end

      -- Plot training curve(s)
        local plots = {{'Reconstruction', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
        plots[#plots + 1] = {'Latent', torch.linspace(1, #latentlosses, #latentlosses), torch.Tensor(latentlosses), '-'}
        plots[#plots + 1] = {'Adversary', torch.linspace(1, #advlosses, #advlosses), torch.Tensor(advlosses), '-'}
        plots[#plots + 1] = {'AdversaryGen', torch.linspace(1, #advGenLosses, #advGenLosses), torch.Tensor(advGenLosses), '-'}
        plots[#plots + 1] = {'MinimaxAdvLatent', torch.linspace(1, #advMinimaxLoss, #advMinimaxLoss), torch.Tensor(advMinimaxLoss), '-'}
        plots[#plots + 1] = {'MinimaxAdvGen', torch.linspace(1, #advGenMinimaxLoss, #advGenMinimaxLoss), torch.Tensor(advGenMinimaxLoss), '-'}
        plots[#plots + 1] = {'Reencode', torch.linspace(1, #reencodelosses, #reencodelosses), torch.Tensor(reencodelosses), '-'}

        if opt.epoch % opt.saveProgressIter == 0 then
            
            encoder:evaluate()
            decoder:evaluate()
            rotate_tmp = model_opts.rotate
            dataProvider.opts.rotate = false

            local x_in, x_out = dataProvider:getImages(torch.linspace(1,10,10):long(), 'train')
            recon_train = evalIm(x_in,x_out)
    
            local x_in, x_out = dataProvider:getImages(torch.linspace(1,10,10):long(), 'test')
            recon_test = evalIm(x_in,x_out)
            
            local reconstructions = torch.cat(recon_train, recon_test,2)

            image.save(model_opts.save_dir .. '/progress.png', reconstructions)

            embeddings = {}
            embeddings.train = torch.zeros(ndat, model_opts.nLatentDims)
            
            indices = torch.linspace(1,ndat,ndat):long():split(opt.batchSize)

            start = 1
            for t,v in ipairs(indices) do
                collectgarbage()
                stop = start + v:size(1) - 1

                x_in = dataProvider:getImages(v, 'train')
                -- Forward pass
                x_in = x_in:cuda()

                codes = encoder:forward(x_in)
                embeddings.train:sub(start, stop, 1,model_opts.nLatentDims):copy(codes[#codes])
                
                start = stop + 1
            end

            ntest = dataProvider.test.inds:size()[1]
            embeddings.test = torch.zeros(ntest, model_opts.nLatentDims)
            indices = torch.linspace(1,ntest,ntest):long():split(opt.batchSize)
            
            start = 1
            for t,v in ipairs(indices) do
                collectgarbage()
                stop = start + v:size(1) - 1

                x_in = dataProvider:getImages(v, 'test')
                -- Forward pass
                x_in = x_in:cuda()

                codes = encoder:forward(x_in)

               embeddings.test:sub(start, stop, 1,model_opts.nLatentDims):copy(codes[#codes])
                
                start = stop + 1
            end
            
            x_in = nil
            
            dataProvider.opts.rotate = rotate_tmp

            encoder:training()
            decoder:training()          

            torch.save(model_opts.save_dir .. '/progress_embeddings.t7', embeddings, 'binary', false)
            embeddings = nil
            
            torch.save(model_opts.save_dir .. '/plots_tmp.t7', plots, 'binary', false)
            torch.save(model_opts.save_dir .. '/epoch_tmp.t7', opt.epoch, 'binary', false)
        end
            
        if opt.epoch % opt.saveStateIter == 0 then
            print('Saving model.')
            
            -- save the optimizer states
            torch.save(stateAdv_path, utils.table2float(stateAdv), 'binary', false)
            torch.save(stateAdvGen_path, utils.table2float(stateAdvGen), 'binary', false)
            
            torch.save(stateEnc_path, utils.table2float(stateEnc), 'binary', false)
            torch.save(stateDec_path, utils.table2float(stateDec), 'binary', false)
            
            
            -- save the options
            torch.save(opt_path, opt, 'binary', false)
            torch.save(optEnc_path, optEnc, 'binary', false)
            torch.save(optDec_path, optDec, 'binary', false)
            torch.save(optAdv_path, optAdv, 'binary', false)
            torch.save(optAdvGen_path, optAdvGen, 'binary', false)            
            
            decoder:clearState()
            encoder:clearState()
            adversary:clearState()
            adversaryGen:clearState()            
            
            torch.save(model_opts.save_dir .. '/plots.t7', plots, 'binary', false)
            torch.save(model_opts.save_dir .. '/epoch.t7', opt.epoch, 'binary', false)
            
            torch.save(model_opts.save_dir .. '/decoder.t7', decoder:float(), 'binary', false)
            torch.save(model_opts.save_dir .. '/encoder.t7', encoder:float(), 'binary', false)
            torch.save(model_opts.save_dir .. '/adversary.t7', adversary:float(), 'binary', false)
            torch.save(model_opts.save_dir .. '/adversaryGen.t7', adversaryGen:float(), 'binary', false)
            
            torch.save(model_opts.save_dir .. '/rng.t7', torch.getRNGState(), 'binary', false)
            torch.save(model_opts.save_dir .. '/rng_cuda.t7', cutorch.getRNGState(), 'binary', false)            
            
            decoder:cuda()
            encoder:cuda()
            adversary:cuda()
            adversaryGen:cuda()
            
        end
        
        plots = nil
    end
end

return learner
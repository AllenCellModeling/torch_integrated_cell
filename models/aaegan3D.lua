-- update 1/13/17 - using an advarsarial network that seems to work well
-- update 2/2/17 - using gan settings from previous experiments (caae-nopool.lua)


require 'nn'
require 'dpnn'

local Model = {
  zSize = 16 --  -- Size of isotropic multivariate Gaussian Z
}

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      -- m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end



function Model:create(opts)
    self.ganNoise = opts.ganNoise or 0
    self.ganNoiseAllLayers = opts.ganNoiseAllLayers or false
    
    self.nLatentDims = opts.nLatentDims
    self.nClasses = opts.nClasses
    self.nChIn = opts.nChIn
    self.nChOut = opts.nChOut
    self.nOther = opts.nOther
    
    Model:createAutoencoder()
    Model:createAdversary()
    Model:createGenAdversary()
    
    Model:assemble()
end
    
function Model:createAutoencoder()
    kT, kW, kH = 5, 5, 5
    dT, dW, dH = 2, 2, 2
    padT, padW, padH = 2,2,2
    
    
    dr = self.dropoutRate

    -- Create encoder
    self.encoder = nn.Sequential()
    
    self.encoder:add(nn.VolumetricConvolution(self.nChIn, 64, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.encoder:add(nn.VolumetricBatchNormalization(64))
    self.encoder:add(nn.PReLU())
    
    self.encoder:add(nn.VolumetricConvolution(64, 128, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.encoder:add(nn.VolumetricBatchNormalization(128))
    self.encoder:add(nn.PReLU())
    
    self.encoder:add(nn.VolumetricConvolution(128, 256, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.encoder:add(nn.VolumetricBatchNormalization(256))
    self.encoder:add(nn.PReLU())
    
    self.encoder:add(nn.VolumetricConvolution(256, 512, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.encoder:add(nn.VolumetricBatchNormalization(512))
    self.encoder:add(nn.PReLU())
    
    self.encoder:add(nn.VolumetricConvolution(512, 512, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.encoder:add(nn.View(512*2*4*3))
    self.encoder:add(nn.BatchNormalization(512*2*4*3))
    self.encoder:add(nn.PReLU())
    
    encoder_out = nn.ConcatTable()
    
    if self.nClasses > 0 then
        -- output for the class label
        pred_label = nn.Sequential()

        pred_label:add(nn.Linear(512*2*4*3, self.nClasses))
        pred_label:add(nn.BatchNormalization(self.nClasses))
        pred_label:add(nn.LogSoftMax())
            
        encoder_out:add(pred_label)
    end
    
    if self.nOther > 0 then
    -- output for the structure features
        pred_other = nn.Sequential()

        pred_other:add(nn.Linear(512*2*4*3, self.nOther))
        pred_other:add(nn.BatchNormalization(self.nOther))
        
        encoder_out:add(pred_other) 
    end
            
    if self.nLatentDims > 0 then
        -- output for the noise
        noise = nn.Sequential()

        noise:add(nn.Linear(512*2*4*3, self.nLatentDims))
        noise:add(nn.BatchNormalization(self.nLatentDims))
        
        encoder_out:add(noise)
    end
    -- join them all together              
    self.encoder:add(encoder_out)                                                 
                            
                    
    -- Create decoder
    self.decoder = nn.Sequential()
    self.decoder:add(nn.JoinTable(-1))
    self.decoder:add(nn.Linear(self.nClasses + self.nOther + self.nLatentDims, 512*2*4*3))

    self.decoder:add(nn.View(512,2,4,3))
    self.decoder:add(nn.VolumetricBatchNormalization(512))    
    self.decoder:add(nn.PReLU())
    
    self.decoder:add(nn.VolumetricFullConvolution(512, 512, kT-1, kW-1, kH-1, dT, dW, dH, padT-1, padW-1, padH-1))
    self.decoder:add(nn.VolumetricBatchNormalization(512))
    self.decoder:add(nn.PReLU())
    
    self.decoder:add(nn.VolumetricFullConvolution(512, 256, kT-1, kW-1, kH-1, dT, dW, dH, padT-1, padW-1, padH-1))
    self.decoder:add(nn.VolumetricBatchNormalization(256))
    self.decoder:add(nn.PReLU())
    
    self.decoder:add(nn.VolumetricFullConvolution(256, 128, kT-1, kW-1, kH-1, dT, dW, dH, padT-1, padW-1, padH-1))
    self.decoder:add(nn.VolumetricBatchNormalization(128))
    self.decoder:add(nn.PReLU())
    
    self.decoder:add(nn.VolumetricFullConvolution(128, 64, kT-1, kW-1, kH-1, dT, dW, dH, padT-1, padW-1, padH-1))
    self.decoder:add(nn.VolumetricBatchNormalization(64))
    self.decoder:add(nn.PReLU())
    
    self.decoder:add(nn.VolumetricFullConvolution(64, self.nChOut, kT-1, kW-1, kH-1, dT, dW, dH, padT-1, padW-1, padH-1))
    -- this being commented out is necessary for reasonable convergence (???)
    -- self.decoder:add(nn.VolumetricBatchNormalization(self.nChOut))
    self.decoder:add(nn.Sigmoid(true))

    self.encoder:apply(weights_init)
    self.decoder:apply(weights_init)
end

function Model:assemble()
    self.autoencoder = nn.Sequential()
    self.autoencoder:add(self.encoder)
    self.autoencoder:add(self.decoder)    
end

function Model:createAdversary()

    self.adversary = nn.Sequential()
      
    self.adversary:add(nn.Linear(self.nLatentDims, 1024))
    self.adversary:add(nn.LeakyReLU(0.2, true))
    
    self.adversary:add(nn.Linear(1024, 1024))
    self.adversary:add(nn.BatchNormalization(1024))
    self.adversary:add(nn.LeakyReLU(0.2, true))
    
    self.adversary:add(nn.Linear(1024, 512)) 
    self.adversary:add(nn.BatchNormalization(512))
    self.adversary:add(nn.LeakyReLU(0.2, true))
    
    self.adversary:add(nn.Linear(512, 1))
    self.adversary:add(nn.Sigmoid(true))

    self.adversary:apply(weights_init)

end

function Model:createGenAdversary()
    noise = self.ganNoise
    ndf = 64
    
    kT, kW, kH = 5, 5, 5
    dT, dW, dH = 2, 2, 2
    padT, padW, padH = 2, 2, 2
    
    self.adversaryGen = nn.Sequential()

    if noise > 0 then
        self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end
    
    self.adversaryGen:add(nn.VolumetricConvolution(self.nChOut, ndf, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.adversaryGen:add(nn.VolumetricBatchNormalization(ndf))  
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    self.adversaryGen:add(nn.VolumetricConvolution(ndf, ndf * 2, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.adversaryGen:add(nn.VolumetricBatchNormalization(ndf * 2))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    self.adversaryGen:add(nn.VolumetricConvolution(ndf * 2, ndf * 4, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.adversaryGen:add(nn.VolumetricBatchNormalization(ndf * 4))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    self.adversaryGen:add(nn.VolumetricConvolution(ndf * 4, ndf * 8, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.adversaryGen:add(nn.VolumetricBatchNormalization(ndf * 8))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    self.adversaryGen:add(nn.VolumetricConvolution(ndf * 8, ndf * 8, kT, kW, kH, dT, dW, dH, padT, padW, padH))
    self.adversaryGen:add(nn.VolumetricBatchNormalization(ndf * 8))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    if self.nClasses > 0 then
        -- output for the class label
        self.adversaryGen:add(nn.VolumetricConvolution(ndf * 8, self.nClasses+1, 2,3, 4))
        self.adversaryGen:add(nn.View(self.nClasses+1))
        -- self.adversaryGen:add(nn.BatchNormalization(self.nClasses+1))
        self.adversaryGen:add(nn.LogSoftMax())
    else
        self.adversaryGen:add(nn.VolumetricConvolution(ndf * 8, 1, 2,3,4))
        self.adversaryGen:add(nn.View(1))
        -- self.adversaryGen:add(nn.BatchNormalization(1))
        self.adversaryGen:add(nn.Sigmoid())
    end
    
    self.adversaryGen:apply(weights_init)
end


return Model




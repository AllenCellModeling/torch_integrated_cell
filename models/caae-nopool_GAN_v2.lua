-- updated with apparently working advarsarial network

require 'nn'
require 'dpnn'

local Model = {
  zSize = 16 --  -- Size of isotropic multivariate Gaussian Z
}

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


function Model:create(opts)
    self.scale_factor = 8/(512/opts.imsize)
    self.ganNoise = opts.ganNoise or 0.01
    self.ganNoiseAllLayers = opts.ganNoiseAllLayers or false
    
    self.nLatentDims = opts.nLatentDims
    self.nClasses = opts.nClasses
    self.nChIn = opts.nChIn
    self.nChOut = opts.nChOut
    self.nOther = opts.nOther
    
    
    Model:createAutoencoder()
    Model:createAdversary()
    Model:createAdversaryGen()
    
    Model:assemble()
end
    
function Model:createAutoencoder()
    
    scale = self.scale_factor
    
    -- Create encoder
    self.encoder = nn.Sequential()
    
    self.encoder:add(nn.SpatialConvolution(self.nChIn, 64, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(64))
    
    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(128))

    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(256))

    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(512))
    
    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(512, 1024, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(1024))
    
    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(1024, 1024, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(1024))
    
    self.encoder:add(nn.View(1024*scale*scale))  
    self.encoder:add(nn.PReLU())
    
    encoder_out = nn.ConcatTable()
        
    if self.nClasses > 0 then
        -- output for the class label
        pred_label = nn.Sequential()
        pred_label:add(nn.Linear(1024*scale*scale, self.nClasses))
        pred_label:add(nn.BatchNormalization(self.nClasses))
        pred_label:add(nn.LogSoftMax())
            
        encoder_out:add(pred_label)
    end
    
    if self.nOther > 0 then
    -- output for the structure features
        pred_other = nn.Sequential()
        pred_other:add(nn.Linear(1024*scale*scale, self.nOther))
        pred_other:add(nn.BatchNormalization(self.nOther))
        
        encoder_out:add(pred_other) 
    end
    
    if self.nLatentDims > 0 then
        -- output for the noise
        noise = nn.Sequential()  
        noise:add(nn.Linear(1024*scale*scale, self.nLatentDims))
        noise:add(nn.BatchNormalization(self.nLatentDims))
        
        encoder_out:add(noise)
    end
            
    -- join them all together              
    self.encoder:add(encoder_out)                                                 
                    
    ngf = 64
    
    -- Create decoder
    self.decoder = nn.Sequential()
    self.decoder:add(nn.JoinTable(-1))
    self.decoder:add(nn.Linear(self.nClasses + self.nOther + self.nLatentDims, 1024*scale*scale))
    self.decoder:add(nn.View(1024, scale, scale))
    
    self.decoder:add(nn.PReLU())    
    self.decoder:add(nn.SpatialFullConvolution(1024, 1024, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(1024))
    
    self.decoder:add(nn.PReLU())    
    self.decoder:add(nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(512))
    
    self.decoder:add(nn.PReLU())    
    self.decoder:add(nn.SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(256))
    
    self.decoder:add(nn.PReLU())      
    self.decoder:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(128))
    
    self.decoder:add(nn.PReLU())      
    self.decoder:add(nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(64))
    
    self.decoder:add(nn.PReLU())      
    self.decoder:add(nn.SpatialFullConvolution(64, self.nChOut, 4, 4, 2, 2, 1, 1))
    -- self.decoder:add(nn.SpatialBatchNormalization(self.nChOut))    
    self.decoder:add(nn.Sigmoid())
    
    self.encoder:apply(weights_init)
    self.decoder:apply(weights_init)
end

function Model:assemble()
    self.autoencoder = nn.Sequential()
    self.autoencoder:add(self.encoder)
    self.autoencoder:add(self.decoder)    
end

function Model:createAdversary()
    -- also modeled off of soumith dcgan
    noise = 0.1
    
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

function Model:createAdversaryGen()
    noise = self.ganNoise
    ndf = 64
    
    scale = self.scale_factor
    
    self.adversaryGen = nn.Sequential()

    -- input is (nc) x 64 x 64
    if noise > 0 then
        self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end
    self.adversaryGen:add(nn.SpatialConvolution(self.nChOut, ndf, 4, 4, 2, 2, 1, 1))
    -- self.adversaryGen:add(nn.SpatialBatchNormalization(ndf))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))

    if self.ganNoiseAllLayers and noise > 0 then
        self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end
    self.adversaryGen:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    self.adversaryGen:add(nn.SpatialBatchNormalization(ndf * 2))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    if self.ganNoiseAllLayers and noise > 0 then
        self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end
    self.adversaryGen:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    self.adversaryGen:add(nn.SpatialBatchNormalization(ndf * 4))   
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    if self.ganNoiseAllLayers and noise > 0 then
        self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end
    self.adversaryGen:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    self.adversaryGen:add(nn.SpatialBatchNormalization(ndf * 8))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    if self.ganNoiseAllLayers and noise > 0 then
        self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end
    self.adversaryGen:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    self.adversaryGen:add(nn.SpatialBatchNormalization(ndf * 8))
    self.adversaryGen:add(nn.LeakyReLU(0.2, true))
    
    -- state size: (ndf*8) x 4 x 4
   if noise > 0 then
       self.adversaryGen:add(nn.WhiteNoise(0, noise))
    end

    if self.nClasses > 0 then
        self.adversaryGen:add(nn.View(ndf * 8 * scale*2 * scale*2))
        self.adversaryGen:add(nn.Linear(ndf * 8 * scale*2 * scale*2, self.nClasses+1))
        self.adversaryGen:add(nn.LogSoftMax())
    else
        
        self.adversaryGen:add(nn.View(ndf * 8 * scale*2 * scale*2))
        self.adversaryGen:add(nn.Linear(ndf * 8 * scale*2 * scale*2, 1))
        
        self.adversaryGen:add(nn.Sigmoid())
    end
    
    self.adversaryGen:apply(weights_init)
end


return Model



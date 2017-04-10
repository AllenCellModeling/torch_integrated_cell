evalIm = function(x_in, x_out)
    local xHat, latentHat, latentHat_var = decoder:forward(encoder:forward(x_in:cuda()))
        
    xHat = imtools.mat2img(xHat)
    x_out = imtools.mat2img(x_out)
        
    -- Plot reconstructions
    recon = torch.cat(image.toDisplayTensor(x_out, 1, x_in:size(1)), image.toDisplayTensor(xHat, 1, x_in:size(1)), 2)

    return recon
end
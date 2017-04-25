evalIm = function(x_in, x_out, opts)
    local codes = encoder:forward(x_in:cuda())
    local xHat = decoder:forward(codes)

    local myFile = hdf5.open(opts.saveDir .. '/progress.h5', 'w')
    myFile:write('/x_out', x_out:float())
    myFile:write('/xHat', xHat:float())
    myFile:close()
    
    xHat = imtools.im2projection(xHat:float())
    dat_out = imtools.im2projection(x_out:float())

    -- Plot reconstructions
    recon = torch.cat(image.toDisplayTensor(dat_out, 1, dat_out:size()[1]), image.toDisplayTensor(xHat, 1, xHat:size()[1]), 2)
    
    return recon
end


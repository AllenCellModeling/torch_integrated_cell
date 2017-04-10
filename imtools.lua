require 'torch'
require 'image'
require 'paths'
require 'colormap'
require 'utils'

imtools = {}

function imtools.load_img(im_dir, im_pattern, im_scale)
    if im_scale == nil then
        im_scale = 1;
    end

    local p = {}
    local c = 1
    
    for f in paths.files(im_dir, im_pattern) do
        p[c] = im_dir .. f
        c = c+1
    end
    
    -- natural sort
    local p = utils.alphanumsort(p)
    
    local im_tmp = image.load(p[1], 3, 'double')

    local im_size_tmp = im_tmp:size()

    local im_size = torch.LongStorage(4)
    
    im_size[1] = #p
    im_size[2] = im_size_tmp[1]
    im_size[3] = torch.round(im_size_tmp[2]/im_scale)
    im_size[4] = torch.round(im_size_tmp[3]/im_scale)
    
    local im_out = torch.Tensor(im_size)
    
    for i = 1,im_size[1] do
        im_out[i] = image.scale(image.load(p[i], 3, 'double'), im_size[3], im_size[4], 'bilinear')
    end
    
    return im_out, p
end


function imtools.mat2img(img)
    if img:size(2) < 3 then
        local padDims = 3-img:size(2)
        img = torch.cat(img, torch.zeros(img:size(1), padDims, img:size(4), img:size(4)):typeAs(img), 2)
    end
    return img
end


function imtools.otsu(img)
--     Shamelessley adapted from
--     https://en.wikipedia.org/wiki/Otsu%27s_method
--     grj 4/28/16
    
    local nbins = 256
    
    img = img - torch.min(img);
    img = torch.div(img, torch.max(img)) * (nbins-1)
    
    local counts = torch.histc(img, nbins, 0, nbins)
    p = counts / torch.sum(counts)
    
    local sigma_b = torch.LongTensor(nbins-1)
    
    for t = 1,(nbins-1) do
        local q_L = torch.sum(p[{{1, t}}])
        local q_H = torch.sum(p[{{t + 1, -1}}])

        local mul = torch.cmul( p[{{1,t}}],  torch.linspace(1, t, t))
        local miu_L = torch.sum( mul ) / q_L;
        
        local muh = torch.cmul(p[{{t+1, -1}}],torch.linspace(t+1, nbins, nbins-t))
        local miu_H = torch.sum(muh) / q_H
        
        sigma_b[t] = q_L * q_H * ((miu_L - miu_H)^2)
    end

    local y, i = torch.max(sigma_b, 1)

    local i = torch.FloatTensor(1):copy(i)
    
    local im2 = img:gt(i[1]);
    return im2
end

function imtools.im2projection(im)
    local im_out = nil
    if im:size():size() == 5 then
        im_out = torch.zeros(im:size()[1], 3, im:size()[4], im:size()[5]):typeAs(im)
        for i = 1,im:size()[1] do
            im_out[i] = imtools.im2projection(im[i])
        end
    else
        local imsize = im:size()
        local nchan = imsize[1]

        colormap:setStyle('jet')
        colormap:setSteps(nchan)
        local colors = colormap:colorbar(nchan, 2)[{{},{},{1}}]

        if nchan == 3 then
            colors = colors:index(2, torch.LongTensor{3,1,2})
        end

        local im_flat = torch.max(im,2)

        im_out = torch.zeros(nchan, 3, imsize[3], imsize[4]):typeAs(im)

        for i = 1,nchan do
            local im_chan = im_flat[i]
            local im_chan = torch.repeatTensor(im_chan, 3, 1, 1)
            local cmap = torch.repeatTensor(colors[{{},{i}}], 1, imsize[3], imsize[4]):typeAs(im_chan)
            
            im_out[{i}] = torch.cmul(im_chan, cmap)
            
            local im_max = torch.max(im_out[{i}])
            if im_max ~= 0 then
                im_out[{i}] = torch.div(im_out[{i}], im_max)
            end
        end

        im_out = torch.squeeze(torch.sum(im_out, 1))
        
        local im_max = torch.max(im_out)
        if im_max ~= 0 then
            im_out = torch.div(im_out, im_max)
        end
    end
    -- print(im_out)
    return im_out
end

    
"""Forward-only configurable tiled renderer. Tensor operations stay on input device.

This deliberately does not claim parity or performance equivalence with Spark or
Ascend fused kernels. Projection is chunked; blending uses at most 128 x 256
Gaussian/pixel pairs per chunk. No model tensors are modified in place.
"""
import math
import re

DEFAULTS = dict(background='#000000', sort='depth', alpha=1.0, opacity=1.0,
                coverage=3.33, minRadius=0.0, maxRadius=512.0,
                preBlur=0.0, blur=0.3, clip=1.3, focal=1.0, near=0.01, far=200.0)
RANGES = dict(alpha=(0, 12), opacity=(0, 2), coverage=(1, 4), minRadius=(0, 10),
              maxRadius=(16, 1024), preBlur=(0, 2), blur=(0, 2), clip=(1, 2),
              focal=(0.5, 2), near=(0.001, 10), far=(1, 100000))


def options(data):
    mode = data.get('renderer', 'native')
    if mode not in ('native', 'tunable'):
        raise ValueError('渲染模式无效')
    raw = data.get('renderOptions', {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('渲染参数包含未知字段')
    result = {**DEFAULTS, **raw}
    for key, (low, high) in RANGES.items():
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} 必须为 {low}–{high} 内的有限数值')
    if result['minRadius'] > result['maxRadius'] or result['near'] >= result['far']:
        raise ValueError('最小半径或近裁剪面必须小于对应上限')
    if result['sort'] not in ('depth', 'radial'):
        raise ValueError('排序方式无效')
    if not isinstance(result['background'], str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', result['background']):
        raise ValueError('背景色必须是 #RRGGBB')
    if mode == 'native' and result != DEFAULTS:
        raise ValueError('自定义参数需要选择可调 NPU 模式')
    return mode, result


def render(splats, camera, width, height, fov, degree, settings, evaluate_sh):
    import torch
    p = settings
    device = splats['means'].device
    bg = torch.tensor([int(p['background'][i:i+2], 16)/255 for i in (1, 3, 5)], device=device)
    focal = height * 0.5 / math.tan(math.radians(fov)*0.5) * p['focal']
    rotation = camera[:3, :3].T
    eye = camera[:3, 3]
    packs = []
    # Bound temporary 3D covariance/Jacobian storage for million-Gaussian scenes.
    for start in range(0, splats['means'].shape[1], 65536):
        stop = start + 65536
        means = splats['means'][:, start:stop].T
        delta = means - eye
        xyz = delta @ rotation.T
        z = xyz[:, 2]
        safe_z = z.clamp_min(p['near'])
        xy = xyz[:, :2] / safe_z[:, None]
        centers = xy * focal + torch.tensor([width/2, height/2], device=device)
        q = torch.nn.functional.normalize(splats['quats'][:, start:stop].T, dim=-1)
        w, x, y, zz = q.unbind(-1)
        r = torch.stack((1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y),
                         2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x),
                         2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)), -1).reshape(-1,3,3)
        transform = rotation[None] @ (r * splats['scales'][:, start:stop].T.exp()[:, None, :])
        clamped = torch.stack((xy[:, 0].clamp(-p['clip']*width/(2*focal), p['clip']*width/(2*focal)),
                               xy[:, 1].clamp(-p['clip']*height/(2*focal), p['clip']*height/(2*focal))), -1)
        j = torch.zeros((len(z),2,3), device=device)
        j[:,0,0] = focal/safe_z
        j[:,1,1] = focal/safe_z
        j[:,:,2] = -clamped*focal/safe_z[:,None]
        projected = j @ transform
        cov = projected @ projected.transpose(1,2)
        a, b, c = cov[:,0,0]+p['preBlur'], cov[:,0,1], cov[:,1,1]+p['preBlur']
        det_before = (a*c-b*b).clamp_min(1e-12)
        a, c = a+p['blur'], c+p['blur']
        det = (a*c-b*b).clamp_min(1e-12)
        opacity = splats['opacities'][start:stop].sigmoid() * p['opacity']
        # Antialiasing preserves Gaussian integral when covariance is enlarged.
        opacity = (opacity * (det_before/det).sqrt()).clamp(0, .999)
        eigenmax = .5*(a+c+((a-c).square()+4*b*b).sqrt())
        radius = p['coverage']*eigenmax.clamp_min(1e-12).sqrt()
        scale = (p['maxRadius']/radius.clamp_min(1e-12)).clamp(max=1).square()
        a, b, c = a*scale, b*scale, c*scale
        det = (a*c-b*b).clamp_min(1e-12)
        radius = radius.clamp(max=p['maxRadius'])
        valid = ((z > p['near']) & (z < p['far']) & (radius >= p['minRadius'])
                 & (opacity >= p['alpha']/255) & (opacity > 0)
                 & (centers[:,0]+radius >= 0) & (centers[:,0]-radius < width)
                 & (centers[:,1]+radius >= 0) & (centers[:,1]-radius < height))
        directions = torch.nn.functional.normalize(delta, dim=-1).T[None].contiguous()
        coeffs = splats['sh'][:(degree+1)**2,:,start:stop][None].contiguous()
        colors = (evaluate_sh(degree, directions, coeffs)[0].T + .5).clamp_min(0)
        key = z if p['sort'] == 'depth' else delta.square().sum(-1)
        packed = torch.cat((centers, torch.stack((c/det,-b/det,a/det),-1),
                            opacity[:,None], colors, radius[:,None], key[:,None]),-1)
        valid &= torch.isfinite(packed).all(-1)
        packs.append(packed[valid])
    points = torch.cat(packs)
    points = points[torch.argsort(points[:,-1])]
    image = bg.expand(height,width,3).clone()
    for top in range(0,height,16):
        for left in range(0,width,16):
            h, w = min(16,height-top), min(16,width-left)
            selected = ((points[:,0]+points[:,9] >= left) & (points[:,0]-points[:,9] <= left+w-1)
                        & (points[:,1]+points[:,9] >= top) & (points[:,1]-points[:,9] <= top+h-1))
            tile = points[selected]
            ys, xs = torch.meshgrid(torch.arange(top,top+h,device=device), torch.arange(left,left+w,device=device), indexing='ij')
            xs, ys = xs.flatten(), ys.flatten()
            trans = torch.ones(h*w,device=device)
            rgb = torch.zeros(h*w,3,device=device)
            for offset in range(0,len(tile),128):
                batch = tile[offset:offset+128]
                dx, dy = xs[None]-batch[:,0,None], ys[None]-batch[:,1,None]
                distance = batch[:,2,None]*dx*dx+2*batch[:,3,None]*dx*dy+batch[:,4,None]*dy*dy
                alpha = (batch[:,5,None]*torch.exp(-.5*distance.clamp_min(0))).clamp(max=.999)
                alpha = torch.where((distance <= p['coverage']**2) & (alpha >= p['alpha']/255), alpha, 0.)
                remain = torch.cumprod(1-alpha,dim=0)
                before = torch.cat((torch.ones_like(remain[:1]),remain[:-1]),dim=0)
                weights = trans[None]*before*alpha
                rgb += weights.T @ batch[:,6:9]
                trans *= remain[-1]
            image[top:top+h,left:left+w] = (rgb + trans[:,None]*bg).reshape(h,w,3)
    return image[None]

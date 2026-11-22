"""CPU mathematical checks for the same tensor renderer used on NPU."""
import unittest
import torch
from tunable import DEFAULTS, options, render


def colors(degree, directions, coefficients):
    return coefficients[:, 0]  # Synthetic post-SH colors, with the +0.5 offset below.


def scene(means, rgb, scales=None, opacity=None):
    n = len(means)
    return dict(means=torch.tensor(means, dtype=torch.float32).T,
                quats=torch.tensor([[1.,0,0,0]]*n).T,
                scales=torch.tensor(scales or [[.1,.1,.1]]*n).log().T,
                opacities=torch.logit(torch.tensor(opacity or [.5]*n)),
                sh=(torch.tensor(rgb, dtype=torch.float32)-.5).T[None])


def draw(model, **settings):
    return render(model, torch.eye(4), 32, 32, 60, 0,
                  {**DEFAULTS, 'blur': 0, **settings}, colors)[0]


class ParameterTests(unittest.TestCase):
    def test_background_alpha_and_compositing(self):
        model = scene([[0,0,2]], [[1,0,0]])
        result = draw(model, background='#0000ff')
        torch.testing.assert_close(result[16,16], torch.tensor([.5,0,.5]))
        torch.testing.assert_close(result[0,0], torch.tensor([0.,0,1.]))
        torch.testing.assert_close(draw(model, opacity=0, background='#ffffff'), torch.ones(32,32,3))
        self.assertEqual(draw(model, near=3).sum().item(), 0)
        self.assertEqual(draw(model, far=1).sum().item(), 0)
        faint = scene([[0,0,2]], [[1,1,1]], opacity=[.01])
        self.assertGreater(draw(faint, alpha=0).sum().item(), 0)
        self.assertEqual(draw(faint, alpha=12).sum().item(), 0)

    def test_radius_filter_and_coverage(self):
        model = scene([[0,0,2]], [[1,1,1]])
        self.assertGreater(draw(model, coverage=4).sum(), draw(model, coverage=1).sum())
        self.assertEqual(draw(model, minRadius=10).sum().item(), 0)
        large = scene([[0,0,2]], [[1,1,1]], scales=[[2,2,2]])
        self.assertGreater(draw(large, maxRadius=512).sum(), draw(large, maxRadius=16).sum())
        self.assertFalse(torch.allclose(draw(model, blur=1),draw(model)))
        self.assertFalse(torch.allclose(draw(model, preBlur=1),draw(model)))

    def test_sort_focal_and_clip(self):
        model = scene([[2,0,2], [0,0,2.5]], [[1,0,0],[0,0,1]], scales=[[2,2,2]]*2)
        self.assertFalse(torch.allclose(draw(model, sort='radial'),draw(model, sort='depth')))
        self.assertFalse(torch.allclose(draw(model, focal=.5),draw(model)))
        self.assertFalse(torch.allclose(draw(model, clip=1),draw(model, clip=2)))

    def test_chunked_blending(self):
        model = scene([[0,0,2]]*257, [[1,0,0]]*257, opacity=[.01]*257)
        expected = 1-.99**257
        self.assertAlmostEqual(draw(model)[16,16,0].item(),expected,places=5)

    def test_validation_and_immutability(self):
        for patch in [{'blur':float('nan')},{'coverage':0},{'background':'red'},
                      {'sort':'bad'},{'near':5,'far':2},{'focal':True},{'unknown':1}]:
            with self.assertRaises(ValueError): options({'renderer':'tunable','renderOptions':patch})
        with self.assertRaises(ValueError): options({'renderOptions':{'opacity':0}})
        model=scene([[0,0,2]],[[1,0,0]])
        original={k:v.clone() for k,v in model.items()}
        draw(model, opacity=2, blur=1, focal=2)
        for key in model: torch.testing.assert_close(model[key],original[key])


if __name__ == '__main__': unittest.main()

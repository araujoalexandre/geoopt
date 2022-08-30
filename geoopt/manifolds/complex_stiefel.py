import torch
from typing import Union, Tuple, Optional
from .base import Manifold
from .. import linalg
from ..utils import size2shape
from ..tensor import ManifoldTensor


__all__ = ["ComplexStiefel", "ComplexCanonicalStiefel"]


_stiefel_doc = r"""
    Manifold induced by the following matrix constraint:

    .. math::

        \hat{X}^\top X = I\\
        X \in \mathrm{C}^{n\times m}\\
        n \ge m
"""


class ComplexStiefel(Manifold):
    __doc__ = r"""
    {}
    Parameters
    ----------
    canonical : bool
        Use canonical inner product instead of euclidean one (defaults to canonical)
    See Also
    --------
    :class:`CanonicalStiefel`, :class:`EuclideanStiefel`, :class:`EuclideanStiefelExact`
    """.format(
        _stiefel_doc
    )
    ndim = 2

    def __new__(cls):
        if cls is ComplexStiefel:
            return super().__new__(ComplexCanonicalStiefel)
        else:
            return super().__new__(cls)

    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] <= shape[-2]
        if not shape_is_ok:
            return (
                False,
                "`{}` should have shape[-1] <= shape[-2], got {} </= {}".format(
                    name, shape[-1], shape[-2]
                ),
            )
        return True, None

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x)
        U, _, V = linalg.svd(x, full_matrices=False)
        out = torch.einsum("...ik,...kj->...ij", U, V)
        return torch.view_as_real(out)

    def random_naive(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Naive approach to get random matrix on Stiefel manifold.
        A helper function to sample a random point on the Stiefel manifold.
        The measure is non-uniform for this method, but fast to compute.
        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device
        Returns
        -------
        ManifoldTensor
            random point on Stiefel manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        tens = torch.randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(linalg.qr(tens)[0], manifold=self)

    random = random_naive

    def origin(self, *size, dtype=None, device=None, seed=42) -> torch.Tensor:
        """
        Identity matrix point origin.
        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored
        Returns
        -------
        ManifoldTensor
        """
        self._assert_check_shape(size2shape(*size), "x")
        eye = torch.zeros(*size, dtype=dtype, device=device)
        eye[..., torch.arange(eye.shape[-1]), torch.arange(eye.shape[-1])] += 1
        return ManifoldTensor(eye, manifold=self)


class ComplexCanonicalStiefel(ComplexStiefel):
    __doc__ = r"""Complex Stiefel Manifold with Canonical inner product
    {}
    """.format(
        _stiefel_doc
    )

    name = "ComplexStiefel(canonical)"
    reversible = True

    @staticmethod
    def _amat(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u @ x.conj().transpose(-1, -2) - x @ u.conj().transpose(-1, -2)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        # <u, v>_x = tr(u^T(I-1/2xx^T)v)
        # = tr(u^T(v-1/2xx^Tv))
        # = tr(u^Tv-1/2u^Txx^Tv)
        # = tr(u^Tv-1/2u^Txx^Tv)
        # = tr(u^Tv)-1/2tr(x^Tvu^Tx)
        # = \sum_ij{(u*v}_ij}-1/2\sum_ij{(x^Tv * x^Tu)_ij}
        u = torch.view_as_complex(u)
        x = torch.view_as_complex(x)
        xtu = x.conj().transpose(-1, -2) @ u
        if v is None:
            xtv = xtu
            v = u
        else:
            v = torch.view_as_complex(v)
            xtv = x.conj().transpose(-1, -2) @ v
        out = (u * v).sum([-1, -2], keepdim=keepdim) - 0.5 * (xtv * xtu).sum(
            [-1, -2], keepdim=keepdim
        )
        return torch.view_as_real(out)

    def _transp_follow_one(
        self, x: torch.Tensor, v: torch.Tensor, *, u: torch.Tensor
    ) -> torch.Tensor:
        a = self._amat(x, u)
        rhs = v + 1 / 2 * a @ v
        lhs = -1 / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qv = torch.linalg.solve(lhs, rhs)
        return qv

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        x = torch.view_as_complex(x)
        u = torch.view_as_complex(u)
        v = torch.view_as_complex(v)
        out = self._transp_follow_one(x, v, u=u)
        return torch.view_as_real(out)

    transp_follow_expmap = transp_follow_retr

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.view_as_complex(x)
        u = torch.view_as_complex(u)
        v = torch.view_as_complex(v)
        xvs = torch.cat((x, v), -1)
        qxvs = self._transp_follow_one(x, xvs, u=u).view(
            x.shape[:-1] + (2, x.shape[-1])
        )
        new_x, new_v = qxvs.unbind(-2)
        return torch.view_as_real(new_x), torch.view_as_real(new_v)

    expmap_transp = retr_transp

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x)
        u = torch.view_as_complex(u)
        out = u - x @ u.conj().transpose(-1, -2) @ x
        return torch.view_as_real(out)

    egrad2rgrad = proju

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x)
        u = torch.view_as_complex(u)
        out = self._transp_follow_one(x, x, u=u)
        return torch.view_as_real(out)

    expmap = retr




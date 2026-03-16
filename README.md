# Deep learning for high-dimensional PDEs on manifold with Neumann and Dirichlet boundary conditions

Authors: Heng Wang (wheng2025@lzu.edu.cn) and Weihua Deng (dengwh@lzu.edu.cn)

Address: School of Mathematics and Statistics, State Key Laboratory of Natural Product Chemistry, Lanzhou University, Lanzhou 730000, China.

## Constraint-Informed BSDE method

![Neural network architecture](./figures/nn.png)

## Environment configuration

python==3.13.5 torch==2.8.0+cu126

## Abstract

The deep backward stochastic differential equation (BSDE) method is a deep learning algorithm for solving high-dimensional partial differential equations (PDEs) in the whole space, which is trained using the sampled trajectories of microscopic stochastic processes; however, generally it is incapable of solving the problems on a manifold or with boundary constraints. To overcome this issue, we propose a strategy compliant with BSDE theory in this paper, termed the Constraint-Informed BSDE method. The key idea of the Constraint-Informed BSDE method is to first construct appropriate stochastic processes based on the physical implications of boundary conditions and manifold constraints, and further derive the BSDEs that the solutions to the corresponding PDEs satisfy. Specifically, we derive the BSDEs satisfied by the solutions of PDEs with Neumann and Dirichlet boundary conditions on multi-fold sphere and in high-dimensional domain. Finally, we demonstrate the performance of the Constraint-Informed BSDE method through extensive numerical experiments. Since there is no need to explicitly incorporate boundary conditions into the loss function, Constraint-Informed BSDE method exhibits sufficient accuracy, stability, and strong robustness.

## Example

We provide some invocation `example` in files with the `".ipynb"` extension, where you can define and modify any of the components in the sample for testing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## Citation

If you use this code in your research, please cite:
```bibtex
@article{Wang2026,
  title = {Deep learning for high-dimensional PDEs on manifold with Neumann and Dirichlet boundary conditions},
  journal = {Journal of Computational Physics},
  volume = {541},
  pages = {114327},
  year = {2026},
  issn = {0021-9991},
  doi = {10.1016/j.jcp.2025.114327}
}
``` -->
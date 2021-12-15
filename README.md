# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

This is the code for the submission 

It is divided into two tasks:
- Task 1 : Approximating Lipschitz continuous functions with neural networks.
- Task 2 : Calculating distances between probability distributions using neural networks and comparing them to Wasserstein distances.

Three different types of architectures are compared:
- ReLU neural networks without any constraints (simpleReLU)
- ReLU neural networks with constraints (bjorckReLU)
- GroupSort neural networks with constraints (bjorckGroupSort)

For Task 1:
One needs to define:
a) The function to approximate. One can choose between the following functions:
    - PWL (piecewise linear function) and the number of domains
    - sinus
    - square
b) The types of the networks:
    - simpleReLU
    - bjorckReLU
    - bjorckGroupSort
c) The depths and widths of the networks.
d) The training mode:
    - training (finite dataset)
    - test (infinite dataset)
e) The scale of the Gaussian noise


For Task 2:
One needs to define:
Regarding the networks
    a) The types of the networks:
        - simpleReLU
        - bjorckReLU
        - bjorckGroupSort
    b) The depths and widths of the networks.

Regarding the mixtures of Gaussians
    a) The underlying dimension of the space (output_dim, by default output_dim=2)
    b) The number of components (output_modes, by default output_dim=4)

For both tasks, run the bash file commands.sh
By default, quantitative results are saved in a npy file in the folder "results/" and figures are later ploted and saved in the folder "figures/".
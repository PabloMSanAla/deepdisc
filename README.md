# Project Name
Deep Learning Classification of Disc Breaks in Galaxy Surface Brightness Profiles Using Euclid Imaging

## Description
This thesis will develop, train, and evaluate a machine-learning framework to automatically classify radial surface brightness profiles of disc galaxies into Type I, Type II, and Type III systems. The project will build on the reference Euclid Q1 disc-break pipeline, which currently applies change-point detection and piecewise parametric modelling. The student will explore data-driven alternatives based on convolutional or transformer-based sequence architectures applied to 1D radial profiles, with the goal of improving classification accuracy, robustness to noise, and computational scalability.

## Development
Use an environment to help us develop this tool.
```bash
# Add installation commands here
conda env create -f environment.yml
conda activate deepdisc
pip install -e .
```

## Status
Let's follow the progress using this [Google slides](https://docs.google.com/presentation/d/1syoZr8R8bxLEs39b9dmKsQAE7UGNpeWWbFFE7Ro0M0w/edit?usp=sharing)


## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the [License Name] - see the LICENSE file for details.
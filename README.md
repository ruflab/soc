# Grounded language learning in the game Settlers of Catan

Check the [resources](#Resources) section for more documents about this work

---

Grounded Language learning is the field which studies the interactions between language and reality. It is expected to be one way to go beyond current language models.

In this work, I explore how to ground language into the game Settlers Of Catan. I'm using a JAVA simulator containings bots to generate game trajectories which I sue to learn a world model.
Finally, I explore how I can use this world model to ground language.

This work (up to tag v0.1) has been done during my 4 months internship at [ANITI](https://aniti.univ-toulouse.fr/index.php/en/).

If you have any question, feel free to contact me on [twitter](https://twitter.com/morgangiraud)

## Requirements
The software was developed and tested on the following 64-bit operating system:
- macOS 10.15.6 (Catalina)
- CentOS

As the development environment, Python 3.7.9 in combination with PyTorch 1.6.0, PyTorch lightning 0.8.5 and Conda  (and much more) was used.

More details can be found by looking at the `environment_darwin.yml` file, the `setup.py` file and the Makefile.

## Install
For development purpose one needs to install all dependencies:
`make install`

Note: This is possible only on OSX

For running the experiment only, one just need to download the needed distribution and install it using:
```bash
unzip soc-0.1.zip
cd soc-0.1
pip install .
```

A full example to run the code on Google Collab can be found [here](https://colab.research.google.com/drive/11lUfPFMNA7uHQNviuAU3p6yN06fZW-vg)

## Tests
```
make test
```

## Run CI checks
```
make ci
```

## Author
[Morgan Giraud](https://github.com/morgangiraud/)

## Resources
- [Annotated slides for the presentation](https://docs.google.com/presentation/d/1MKxizuQflOzxMjbv_sUYUOTsE_oWZ-g02TQzwOFdNtg/edit#slide=id.p)
- [The presentation video](https://youtu.be/OpnSiUJC9Qw)
- [The JAVA simulator](https://github.com/ruflab/StacSettlers)


## License
See the License.md file at the root of this repository.
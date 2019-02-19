# Aircraft state estimation

## The Sequential Monte Carlo (a.k.a. particle filter)

The repository implements a particle filter for aircraft mass and thrust setting estimation base on the point-mass aircraft performance model. It can also be used for other state filtering.

## Required library
- [OpenAP](https://github.com/junzis/openap) (the aircraft performance library)


## Examples
```sh
python simulation.py

python estimate.py --ac B737 --eng CFM56-7B24 --fin data/b737_example_1.csv --noise n3
```

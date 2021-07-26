# I Don't Need u: Identifiable Non-Linear ICA Without Side Information
## Dependencies

This repository contains code to run and reproduce the experiments presented in `I Don't Need u: Identifiable Non-Linear ICA Without Side Information`
## Dependencies

We include a requirements.txt

## Running  experiments

These experiments are run through the script `main.py`. Below are details on how to use the script. To learn about its arguments type `python main.py --help`:

```
usage: main.py [-h] [--config CONFIG] [--run RUN] [--n-sims N_SIMS]
               [--seed SEED] [--baseline] [--representation] [-z]
               [--mcc] [--second-seed SECOND_SEED] [--all] [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --run RUN             Path for saving running related data.
  --n-sims N_SIMS       Number of simulations to run
  --seed SEED           Random seed
  --z dim z             Dimensionality of Latent Space
  --baseline            Run the script for the baseline
  --representation      Run CCA representation validation across multiple
                        seeds
  --mcc                 compute MCCs for representation
                        experiments
  --second-seed SECOND_SEED
                        Second random seed for computing MCC for representation experiments
  --all                 Run transfer learning experiment for many seeds and
                        subset sizes
  --plot                Plot selected experiment for the selected dataset
```

All options and choice of dataset are passed through a configuration file under the `configs` folder.

To run all the image data experiments in the paper, simply run the shell script `experiments.sh`, though this is set up to run all experiments in a sequence on a single GPU, so would take some time to run.

## License
A full copy of the license can be found [here](LICENSE).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.




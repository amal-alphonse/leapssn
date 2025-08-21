# LeAP-SSN

### Synopsis ###

This Python package implements the LeAP-SSN (<ins>Le</ins>venberg–Marquardt <ins>A</ins>daptive <ins>P</ins>roximal <ins>S</ins>emi<ins>s</ins>mooth <ins>N</ins>ewton method) algorithm described in

"A semismooth Newton method with global convergence rates", A. Alphonse, P. Dvurechensky, I. P. A. Papadopoulos, C. Sirotenko (2025), [arXiv](FIXME).

Use the flag --plot to generate the figures.

|Figure|File: examples/|
|:-:|:-:|
|2|[signorini.py](https://github.com/amal-alphonse/leapssn/blob/main/examples/signorini.py)|
|3|[signorini.py](https://github.com/amal-alphonse/leapssn/blob/main/examples/signorini.py)|
|4|[image_restoration.py](https://github.com/amal-alphonse/leapssn/blob/main/examples/image_restoration.py)|
|5|[support_vector_classification.py](https://github.com/amal-alphonse/leapssn/blob/main/examples/support_vector_classification.py)|

### Dependencies and installation ###

The code is written in Python using Firedrake: a finite element solver platform. Firedrake is well documented here: firedrakeproject.org.

First install Firedrake: https://www.firedrakeproject.org/download.html. Make sure to activate the Firedrake venv and run a firedrake-clean!

    source firedrake/firedrake/bin/activate
    firedrake-clean

Then download and pip install the leapssn library:

    git clone git@github.com:amal-alphonse/leapssn.git
    cd leapssn/
    pip3 install .
    cd ../

### Contributors ###

Amal Alphonse (alphonse@wias-berlin.de)

Pavel Dvurechensky (dvureche@wias-berlin.de)

Ioannis P. A. Papadopoulos (papadopoulos@wias-berlin.de)

Clemens Sirotenko (sirotenko@wias-berlin.de)

### Reference ###

The BaseSmoothOracle and OracleCallsCounter Python classes are originally found in
Nikita Doikov's super-newton repository: https://github.com/doikov/super-newton
which has an MIT License.

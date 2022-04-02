<h1 align="center">HistCL</h1>
<p align="center">
    <strong>Histogram Equalisation in OpenCL.</strong>
</p>

HistCL implmentes a local memory histograms which uses a work-group size of
32 and a work-item size of 1 as it uses uchar4 as a datatype to reduce
global memeory access, the local histograms are then moved to a partial
histogram array and then reduced.
A harris inclusive scan is used to get the cumulated histogram, this is then
turned into the lookup table to save memory.
The look up table is then used to equalise the input image, the equalisation
step also uses uchar4 to reduce global memory access.

CMake is used as a build-system so the project remained cross-platform and
that is the only dependency that shall not be provided.

Argument parsing is done by Argh! (https://github.com/adishavit/argh), Image
IO is provided by STB (https://github.com/nothings/stb), I use a custom
OpenCL wrapper to abstract away kernels and device selection.

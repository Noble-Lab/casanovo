# Downstream Analysis and Visualization Tools

Once Casanovo has inferred peptide sequences, several tools are available to analyze and visualize the results.

## Limelight Integration

[Limelight](https://limelight-ms.org/) is a web application for visualizing and sharing DDA proteomics data and results.
It provides tools for viewing the whole data stack, including raw spectra; extracted ion chromatograms (XICs); peptide, protein, and modification views; quality control (QC) views; and more.
For more information, see the [Limelight publication in JPR](https://pubmed.ncbi.nlm.nih.gov/40036265/).

To upload results to Limelight, Casanovo mzTab results files must first be converted to Limelight XML.
A converter for generating Limelight XML from Casanovo results is available, see the [converter documentation](https://github.com/yeastrc/limelight-import-casanovo).
The Limelight XML may be easily uploaded to Limelight using its web interface.
See [the Limelight documentation](https://limelight-ms.readthedocs.io/en/latest/using-limelight/data-upload-guide.html) for more information about uploading data.

## Nextflow Workflow

To simplify the process of setting up and running Casanovo, a dedicated [Nextflow](https://www.nextflow.io/) workflow is available.
In addition to simplifying the installation of Casanovo and its dependencies, the Casanovo Nextflow workflow provides an automated mass spectrometry data pipeline that converts input data files to a Casanovo-compatible format using [msconvert](https://proteowizard.sourceforge.io/tools/msconvert.html), infers peptide sequences using Casanovo, and (optionally) uploads the results to [Limelight](https://limelight-ms.org/).

The workflow can be used on POSIX-compatible (UNIX) systems, Windows using WSL, or on a cloud platform such as AWS.
For more details, refer to the [Casanovo Nextflow Workflow Documentation](https://nf-ms-dda-casanovo.readthedocs.io/).

## Visualizing Casanovo Results With PDV

[PDV](https://github.com/wenbostar/PDV/) provides a graphical interface for inspecting Casanovo outputs.
PDV allows users to load Casanovo result files and visualize annotated experimental spectra alongside predicted peptide fragmentation patterns.

PDV facilitates interactive exploration of results, making it easier to assess the quality of peptide identifications.
For installation instructions and usage details, refer to the [PDV documentation](https://github.com/wenbostar/PDV/wiki/Visualize-Casanovo-result).

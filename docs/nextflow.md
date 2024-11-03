# Casanovo Nextflow Workflow

To simplify the process of setting up and running Casanovo, a dedicated [Nextflow](https://www.nextflow.io/) workflow is available.
In addition to simplifying the installation of Casanovo and its dependencies, the Casanovo Nextflow workflow provides an automated mass spectrometry data pipeline that converts input data files to a Casanovo-compatible format using [msconvert](https://proteowizard.sourceforge.io/tools/msconvert.html), infers peptide sequences using Casanovo, and (optionally) uploads the results to [Limelight](https://limelight-ms.org/) - a platform for sharing and visualizing proteomics results.
The workflow can be used on POSIX-compatible (UNIX) systems, Windows using WSL, or on a cloud platform such as AWS. 
For more details, refer to the [Casanovo Nextflow Workflow Documentation](https://nf-ms-dda-casanovo.readthedocs.io/en/latest/#).
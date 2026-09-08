raw contains the public source archives and frozen source-network snapshots
needed to rebuild the model inputs.

prepared contains the five supplied runtime files loaded by Main.py. Main.py
uses these files directly and never launches preprocessing.

work is a scratch directory used only by prepare_inputs.py. It is normally
empty because successful preprocessing removes intermediate files unless
--keep-work is used.

See the repository README.txt and data_manifest.csv for details.

(time-domain)=
# Time Domain

In this set of Use Case Scenario we work towards creating multi-band light curves from multiple archival and publication resources at scale and classifying and analyzing them with machine learning tools. Tutorials included in this folder are:

1. [light_curve_collector](light_curve_collector.md): This notebook automatically retrieves target positions from the literature and then queries multiple data archives for light curves of those targets.
2. [scale_up](scale_up.md): This notebook builds on the code demonstrated in light_curve_collector and is recommended for >~500 targets. It is able to generate light curves for a large number of targets (500,000+) and provides additional monitoring options.
3. [light_curve_classifier](light_curve_classifier.md): This notebook takes output from light_curve_collector and trains a ML classifier to be able to differentiate amongst the targets based on their light curves.
4. [ML_AGNzoo](ML_AGNzoo.md): This notebook takes output from the light_curve_collector and visualizes/compares different labelled targets on a reduced dimension grid.

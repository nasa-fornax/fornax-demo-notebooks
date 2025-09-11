# Notebook Review Process

This is a requirements document for notebooks contributed to the fornax-demo-notebooks repo by Fornax team members.
When a notebook is deemed "finished" by its authors, it should go through a two part review; a science review and a tech review.
Checklists for those reviews are below.
Checklists have been written and maintained by the distributed Fornax team.
Checklists should be applied within reason to notebooks, as occaisionally there will be reasons for a notebook to not check all the boxes, and that is ok.

Notebooks in this repo have the extra burden of being tutorials for the general public, so they should be easy to follow, well documented, well maintained, and as fool-proof as possible.  This is different from the average working notebook that we all have written in the past, which may be more of a "scratchpad" for our own use, and not intended for public consumption.




## How to choose reviewers?

-  From the pool of people who have successfully compelted a PR in the fornax-demo-notebooks, choose one developer for a tech review and one scientist for the science review, from any team.  Updates and maintenance to notebooks do not require 2 approvals.




## Science Review Checklist

-   Is there a use case in the introduction which motivates the code?  will our community understand this motivation/code?
-   Does the code do what the intro says it is going to do?
-   Is it scientifically accurate?
-   Does it include all three archives HEASARC, MAST, IRSA?\
    -   if not, is that justified?
-   Does it include work linked to a buzzword:
    -   big data, spectroscopy, time domain, forced photometry, cloud
-   Has each NASA archive been given the option to comment on modules for their relevant data access?\
    -   Discourse `staff` channel is one method of contacting people to check for this
-   Is archival data accessed in the most efficient way according to that archive?

## Tech Review Checklist

-   Documentation:
	-   Is every function documented?
   	-   Do all code cells have corresponding narratives/comments?
   	-   Include information about which server type and environment to choose when logging in to Fornax and the notebook's expected runtime given that setup. For example:
        -   "As of 2024 August, this notebook takes about 3 minutes to run to completion on Fornax using Server Type: 'Standard - 8GB RAM/4 CPU' and Environment: 'Default Astrophysics' (image)."
-   Dependencies and imports:
    -   Does the notebook have a corresponding `requirements_<notebook_filename>.txt` file listing all its direct dependencies?
    -   Are all dependencies listed in the requirements file in fact required? Please revisit the list as the notebook evolves.
    -   Is the requirements file used in a commented-out cell in the notebook  with `# %pip install -r <filename>`; and has the notebook no other installation related cells?
    -   Are dependencies kept to a minimum? E.g. no new library introduced for minimal usage while another library that is already a dependency can do the same functionality?
        (e.g. do not introduce pandas to print out a line from a FITS table for which we already need to use astropy for IO; add dependencies when their functionality is unique or required for efficiency, etc.)
-   Notebook execution, error handling, etc.:
	-   Does the notebook run end-to-end, out of the box?
 	-   Are errors handled appropriately, with `try`/`except` statements that are narrow in scope?
	-   Have warnings been dealt with appropriately, preferably by updating the code to avoid them (i.e., not by simply silencing them)?
   	-   no manual table of contents
-   Efficiency:
	-   Is data accessed from the cloud where possible?
	-   Is the code parallelized where possible?
	-   If the notebook is intended to be scaled up, does it do that efficiently?
	-   Is memory usage optimized where possible?
-   Style:
    - Follow suggestions of The Turing Way community on [markdown style](https://book.the-turing-way.org/community-handbook/style) including but not limited to:
        - Write each sentence in a new line (line breaks) to make changes easier to read in PRs
        - Avoid latin abbreviation to avoid failing CI 
-   Cleanup:
	-   Have blocks of code that need to be re-used been turned into functions (rather than being duplicated)?
	-   Have unused libraries been removed from the requirements.txt file and the `import` statements?
	-   Has unused code been removed (e.g., unused functions and commented-out lines)?
   	-   Are code comment lines wrapped so all fit within a max of 90 - 100 characters per line?
   	-   Are code lines reasonably short where possible? some code lines can't easily be wrapped and that is ok
   	-   Do plots use color-blind friendly palettes for plotting? try this [simulator](https://www.color-blindness.com/coblis-color-blindness-simulator/#google_vignette) for visual check
    -   It is fine to use AI to generate code snippets or surrounding text.  AI generated content should be clearly marked as such, and not used in the notebook unless it is well tested.  AI acknowledgement should be in the acknowledgement section at the bottom of the notebook.



```python

```

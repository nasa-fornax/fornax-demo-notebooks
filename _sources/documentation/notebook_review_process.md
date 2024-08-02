# How Does Fornax Review Notebooks?
***

This is a sort of requirements document for notebooks contributed to Fornax by Fornax team members.  When a notebook is deemed "finished" by its authors, it should go through a two part review; a science review and a tech review.  Suggested checklists for those reviews are below.  These should be applied within reason to notebooks, and especially lightly to contributed notebooks. Checklists have been written and maintained by the distributed Fornax team.

For authors: consider these checklists requirements for your code.

## Who should participate in these reviews?
- JK's suggestion is that if everyone on Fornax helps to write the below checklists, then the whole team does not need to be involved in code reviews.  One developer per tech review, one scientist per science review, from any team.
  


## Science Review Checklist
- Is there a use case in the introduction which motivates the code?  will our community understand this motivation/code?
- Does the code do what the intro says it is going to do?
- Is it scientifically accurate?
- Does it include all three archives HEASARC, MAST, IRSA?\
      - if not, is that justified
- Does it include work linked to a buzzword:
	- big data, spectroscopy, time domain, forced photometry, cloud
- Has each NASA archive been given option to comment on modules for their relevant data access?\
  	- TODO: a preferred contact method for each archive should be listed here, ie., archive helpdesk, NN slack channel #fornaxdev-daskhub? ??? 	
	- Is archival data accessed in the most efficient way according to that archive?
## Tech Review Checklist
- Documentation:
	- Is every function documented?
	- Does it follow the style guide? https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
   	- Do all code cells have corresponding narratives/comments?
   	- Include information about runtime on fiducial Fornax server
   	- Include information about which "image" the notebook uses when loggin into Fornax, ie., "Astrophysics default image"
- Notebook execution, error handling, etc.:
	- Does the notebook run end-to-end, out of the box?
 	- Are errors handled appropriately, with `try`/`except` statements that are narrow in scope?
	- Have warnings been dealt with appropriately, preferably by updating the code to avoid them (i.e., not by simply silencing them)?
- Efficiency:
	- Is data accessed from the cloud where possible?
	- Is the code parallelized where possible?
	- If the notebook is intended to be scaled up, does it do that efficiently?
	- Is memory usage optimized where possible? 
- Cleanup:
	- Have blocks of code that need to be re-used been turned into functions (rather than being duplicated)?
	- Have un-used libraries been removed from the requirements.txt file and the `import` statements?
	- Has un-used code been removed (e.g., unused functions and commented-out lines)?
   	- Are comment lines wrapped so all fit within a max of 90 - 100 characters per line?
   	- Are code lines reasonably short where possible? some code lines can't easily be wrapped and that is ok

```python

```

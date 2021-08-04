# OAI

This repo is a collection of scripts for navigating and making sense of the Osteoarthritis Initiative dataset. With
over 9,000 variables tracked for almost 5,000 patients, this project is a medical goldmine.  It also less than trivial
to get a sense of where to start.  The scripts in this repo are an attempt to help.

## Setting up to use this code

Obviously, you can modify the code to whatever setup you prefer, but it can be helpful to know what setup the original
author used (and sometimes why).


## Where to start
While the SAS parsing is reasonably quick, it is still faster and more space efficient to convert the data into
Pandas dataframes and save them as pickle files. Only a few notebooks must be run before others, and they are noted 
below (as well as in the notebooks themselves). 


### Survey the data
Once data files are in place, the next natural step is to sort the 9,000+ variables into bins that make sense. To some
degree the folks who publish that data already have. Unfortunately, the categorization they did was only recorded
in PDF files. 

#### Parse out categories and sub-categories

`Parsing VG_Forms PDF for variable sources and categories.ipynb` parses out this data and saves it in a pandas frame
for quick consumption by other notebooks. It creates:
* `oai_vars_labels_sources.pkl` - A dataframe of all variable names and their descriptions labels as well as
  their source files.
* `oai_vars_categories_subcategories.pkl` - A dataframe of all variable names and the categories and sub-categories
  they are tagged with.

#### Create summary tables

`Creating Per-Visit Tables.ipynb` uses the var categories and labels to create tables showing how many of each kind of
variable was collected during each visit. Output in wikitables for now. 

### Concat data and convert to dataframes

Breaking out the data into some 150 data files is a bit cluttered. `Create OAI Dataframes.ipynb` was a coarse
stab at concatenation and dataframe creation. This will be a work in progress. As the details of different datasets
is explored and more logical dataframe schemas become evident from actual usage, dataframe creation will move to
`Create Data Specific Dataframes.ipynb`. It is a more attentive attempt to create concatenated dataframes. 

# OAI

This repo is a collection of scripts for navigating and making sense of the Osteoarthritis Initiative dataset. With
over 11,000 variables tracked for almost 5,000 patients, this project is a medical goldmine.  It is also less than
trivial to get a sense of where to start.  The scripts in this repo are an attempt to help by making public the
initial code that any researchers would need to write to use the data outside of SAS.

The only request in using this library is that you mention it in the acknowledgements when you publish. Maintaining
this code takes time from research work. Doing so is worth it if more people benefit and contribute. 

## Setting up to use this code

Obviously, you can modify the code to whatever setup you prefer, but it can be helpful to know what setup the original
author used (and sometimes why). 

* Download: `General Information.zip` and `OAICompleteData_SAS.zip`
* Unzip `General Information.zip` in the `data/pdfs/` directory
* Unzip `OAICompleteData_SAS.zip` in `data/structured_data`
* Move all PDF files from `data/structured_data` to `data/pdfs/Data Descriptions/`

If you decide to store your data elsewhere (esp. images), consider creating a softlink from the given
directory to the actual location. This may be faster than searching for where to modify the scripts.

Note: Notebooks were written presuming they will be run from the notebook directory. All paths are relative from that 
directory.

## Where to start
While the SAS parsing is reasonably quick, it is still faster and more space efficient to convert the data into
Pandas dataframes and save them as pickle files. Only a few notebooks must be run before others, and they are noted 
below (as well as in the notebooks themselves). 


### Survey the data
Once data files are in place, the next natural step is to sort the 9,000+ variables into bins that make sense. To some
degree the folks who publish that data already have. Unfortunately, the categorization they did was only recorded
in PDF files. Note that `Exploring Available SAS Metadata.ipynb` is merely exploring issues related to importing the
data from SAS files into Pandas.

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

#### Group variables by labels
For each test and metric gathered in the OAI research, many variables were collected.  Typically, the labels of these
variables start with the same text. `Looking at variable groups.ipynb` bin variables together by shared label 
prefixes.

### Concat data and convert to dataframes

Breaking out the data into some 145 data files is a bit cluttered. `Convert SAS to Dataframes.ipynb` is a
stab at concatenation and dataframe creation. This is a work in progress. As the details of different datasets
is explored and more logical dataframe schemas become evident from actual usage, customizations will be added.
For now, the notebook can convert all but 5 files. Next round improvements are listed in the TODO section.

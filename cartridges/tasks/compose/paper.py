import os
from typing import List
from capsules.context import StructuredContext
from capsules.generate.structs import Context, Section
from capsules.generate.run import BaseContextConfig


class PaperFigure(StructuredContext):
   desc: str
   content: str




class PaperListing(StructuredContext):
   desc: str
   content: str




class PaperSection(StructuredContext):
   desc: str
   content: str




class Paper(StructuredContext):
   main: str
   sections: List[PaperSection]
   figures: List[PaperFigure]
   listings: List[PaperListing]
   appendix: List[PaperSection]




def load_paper_dataset() -> List[str]:


   filepath = "/root/simran/capsules/capsules/tasks/thunderkittens/arXiv-2410.20399v1/"
   sections = f"{filepath}/Sections/"
   figures = f"{sections}/figures/"
   listings = f"{sections}/listings/"
   appendix = f"{sections}/appendix/"
   main_path = f"{filepath}/main.tex"
  
   outputs = []
   num_tokens = 0
   decriptions = []


   # sections
   section_outputs = []
   section_files = os.listdir(sections)
   section_files = [f for f in section_files if os.path.isfile(os.path.join(sections, f))]
   for file in section_files:
       if not file.endswith(".tex"):
           continue
       with open(os.path.join(sections, file), "rb") as f:
           data = f.read()
           data = data.decode("utf-8", errors="ignore")
           section_outputs.append(
               PaperSection(
                   desc=f"Thunderkittens: {file}",
                   content=data,
               )
           )
           outputs.append(
               Section(
                   content=data,
                   desc=file,
               )
           )
           num_tokens += len(data.split())
           decriptions.append(file)


   # figures
   figure_outputs = []
   for file in os.listdir(figures):
       subdir = os.path.join(figures, file)
       for subfile in os.listdir(subdir):
           if not subfile.endswith(".tex"):
               continue
           with open(os.path.join(subdir, subfile), "rb") as f:
               data = f.read()
               data = data.decode("utf-8", errors="ignore")
               figure_outputs.append(
                   PaperFigure(
                       desc=subfile,
                       content=data,
                   )
               )
               outputs.append(Section(
                   content=data,
                   desc=subfile,
               ))
               num_tokens += len(data.split())
               decriptions.append(subfile)


   # listings
   listing_outputs = []
   for file in os.listdir(listings):
       if not file.endswith(".tex"):
           continue
       with open(os.path.join(listings, file), "rb") as f:
           data = f.read()
           data = data.decode("utf-8", errors="ignore")
           listing_outputs.append(
               PaperListing(
                   desc=file,
                   content=data,
               )
           )
           outputs.append(Section(
               content=data,
               desc=file,
           ))
           num_tokens += len(data.split())
           decriptions.append(file)


   # # appendix
   appendix_outputs = []
   for file in os.listdir(appendix):
       if not file.endswith(".tex"):
           continue
       with open(os.path.join(appendix, file), "rb") as f:
           data = f.read()
           data = data.decode("utf-8", errors="ignore")
           appendix_outputs.append(
               PaperSection(
                   desc=file,
                   content=data,
               )
           )
           outputs.append(Section(
               content=data,
               desc=file,
           ))
           num_tokens += len(data.split())
           decriptions.append(file)


   with open(main_path, "rb") as f:
       data = f.read()
       main = data
       main = data.decode("utf-8", errors="ignore")
       decriptions.append("main.tex")
       outputs.append(Section(
           content=main,
           desc="main.tex",
       ))
       num_tokens += len(main.split())




   print(f"Total number of tokens: {num_tokens}")
   breakpoint()


   output_context = Paper(
       main=main,
       sections=section_outputs,
       figures=figure_outputs,
       listings=listing_outputs,
       appendix=appendix_outputs,
   )


   return output_context, outputs




class PaperContextConfig(BaseContextConfig):
   mode: str = "train"


   def instantiate(self) -> Context:


       context, all_context = load_paper_dataset()


       if self.mode == "train":
           return context


       elif self.mode == "icl":
           return Context(
               sections=[
                   Section(
                       name="paper",
                       content=all_context,
                       description="Paper context",
                   )
               ],
               main=all_context,
           )


       return context

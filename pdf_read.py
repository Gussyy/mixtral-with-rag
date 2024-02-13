import pypdf
# creating a pdf reader object
reader = pypdf.PdfReader(r"C:\Users\prath\Downloads\Mechanics of Materials 10th Edition by Russell C. Hibbeler (z-lib.org) (5).pdf")

# print the number of pages in pdf file
print(len(reader.pages))

# print the text of the first page
print(reader.pages[7].extract_text())
print(type(reader.pages[7].extract_text()))

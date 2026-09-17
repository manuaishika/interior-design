# Catalogue data

**The schema lives in [`../CATALOGUE.md`](../CATALOGUE.md). There is one, and
this is not a second one.**

- `../Second-Draft-catalogue.xlsx` — the workbook to send the client. Drop-downs,
  validation, a Read me tab, three example rows.
- `../catalogue-template.csv` — the same columns as a CSV, generated from that
  workbook so the two cannot drift apart.

Put the client's returned file in here as `catalog.csv` and run the importer.

## Two ways to give us the photos

Either column works; a row should use one, not both.

- `image_url` — the photo is already online. Right-click on their site, copy
  image address.
- `image_file` — the photos are a folder on somebody's computer. The filename
  goes in the column, the folder comes with the sheet, and the importer copies
  them in.

A client with a website will do the first. A client with a hard drive full of
JPEGs will do the second, and telling them to build a website first is how a
catalogue never arrives.

## Measurements are millimetres

Whole numbers, no decimals. A 2.4 metre wardrobe is `2400`. Centimetres invite
`1.5` and then somebody has to guess whether that is fifteen millimetres.

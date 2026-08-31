# Demo catalogue

Placeholder rows in `catalog_template.csv` — replace with the client's real
products. Names and prices are illustrative, not real products.

## Why product IDs, not just pictures

A folder of pictures cannot be sold from. The pipeline already returns, for
every piece of furniture it finds, a label and a box:

    { "label": "bed", "bounding_box": {...}, "locked": false }

`label` is what you look up. Without an ID on the other side there is nothing
to look it up *in* — no price to quote, no link to click, no way to answer
"does that sofa actually fit in this room".

So: **IDs are the spine, pictures hang off them.** One row per product, one
image file named after the ID.

## The fields that matter, and why

| Field | Why it earns its place |
| --- | --- |
| `product_id` | The join key. Everything else hangs off it. |
| `name` | What a person reads. |
| `category` | Matches the pipeline's label (`bed`, `sofa`, `table`, `rug`). |
| `style` | Lets you filter the catalogue to the chosen look. |
| `width/depth/height_cm` | **The one clients actually care about.** A design that recommends a 240cm sofa for a 200cm wall is worthless, and dimensions are what let you check. |
| `price`, `currency` | Turns a render into a quote. |
| `image_file` | For showing the product beside the render. |
| `product_url` | Where to buy it. Leave blank if there isn't one yet. |

## Size to aim for

Thirty to fifty products across five or six categories is plenty for a demo.
Enough that a room can be furnished several different ways; small enough that
you can photograph and measure them all in an afternoon.

## Layout

    data/
      catalog.csv
      images/
        sofa-001.jpg
        bed-001.jpg

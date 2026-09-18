# The catalogue

## What this is for

Right now a render invents furniture and the costing table estimates what it
might cost. With a catalogue, both stop guessing: the design is steered towards
pieces that actually exist, and the price beside it is the real price of a real
thing somebody can order.

## How to collect it — a spreadsheet, with links to the pictures

The confusion is always the images. **You do not put pictures inside a
spreadsheet.** One row per product, and the photo is one of two columns —
`image_url` if it is already online, `image_file` if it is not. That is how
every real product feed works, IKEA's included.

`catalogue-template.csv` in this repo is the template. Send that file to the
client and ask them to fill it in and send it back as CSV.

### The two ways to give us a photo

A row uses one of these, not both — see `data/README.md`.

- **`image_url`** — the photo is already online, on the client's own site or
  anywhere else public. Right-click it, copy the image address, paste that.
- **`image_file`** — the photos are a folder of JPEGs on somebody's computer.
  Just the filename goes in the sheet (`hale-wardrobe.jpg`); the folder
  itself comes alongside it, and
  `python -m app.import_catalogue data/catalog.csv path/to/that/folder`
  copies every file the sheet names into `data/catalog-images/`, which the
  app serves at `/catalog-images`. Telling a client with a hard drive full of
  JPEGs to build a website first is how a catalogue never arrives.

### The columns

| column | required | what it is |
| --- | --- | --- |
| `sku` | **yes** | their own code, unique. The one thing that must never change |
| `name` | **yes** | what a customer would call it |
| `category` | **yes** | from the fixed list below — this is what matching keys off |
| `price` | **yes** | a number, no symbols or commas |
| `currency` | **yes** | `INR`, `GBP`, `USD` |
| `width_mm` `depth_mm` `height_mm` | **yes** | millimetres, whole numbers. Without these it cannot tell what fits |
| `colour` | yes | plain words: `walnut`, `oatmeal`, `charcoal` |
| `material` | yes | plain words |
| `style_tags` | yes | comma-separated, from the six looks the app already has |
| `room_tags` | yes | semicolon-separated, matching the room list |
| `image_url` | one of these two | a direct link to the image file, publicly reachable |
| `image_file` | one of these two | just the filename — see "the two ways to give us a photo" below |
| `product_url` | yes | the page a customer would buy from |
| `in_stock` | yes | `yes` / `no` |
| `lead_time_days` | no | whole number |
| `notes` | no | anything else |

`category` must be one of: `bed`, `sofa`, `chair`, `table`, `desk`, `wardrobe`,
`storage`, `lighting`, `rug`, `soft-furnishing`, `decor`, `appliance`.

That fixed list is the whole point of the column — it is what lets "the desk in
this room" find the desks. A free-text category matches nothing.

### Telling the client

> Please fill in this spreadsheet, one row per product, and send it back as a
> CSV. For the photos: if they are on your website, right-click each picture,
> copy the image address, and paste that into `image_url`. If they are only
> on a computer somewhere, put the filename in `image_file` instead and send
> the photos themselves as a folder alongside the sheet — no website needed.
>
> The three measurements matter more than anything else: without the size in
> millimetres we cannot tell whether a piece fits the room.

Start with fifty rows across every category rather than two thousand of one.
Breadth is what makes a demo work; depth can come later.

## How it plugs in

The seam already exists. Every item the reader finds comes back with a name, a
category and a bounding box — which is exactly what a product lookup needs.

```
read the room      -> "a desk, a wardrobe, a bed"
        |
match the catalogue-> desks in this style, for this room, that fit the space
        |
        +-> into the prompt:  "a walnut writing desk, 1200mm wide"
        +-> beside the render: the real product, its price, its link
        +-> into the costing:  real prices instead of an estimate
```

### The honest limit

Text-prompted generation can be steered *towards* a product but cannot
reproduce a specific SKU faithfully. The render will show something very like
the Fenn desk; it will not be a photograph of the Fenn desk.

So the render is the mood and the product list beside it is the orderable
truth. Say that plainly in the interface rather than letting somebody assume
the picture is the product — that assumption ends in a complaint.

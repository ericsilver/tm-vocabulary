# Rating rubric: product segments

You will rate 60 product segments, each described by the themes its US trademark registrations draw on and by ten example goods/services descriptions (file `segment_packet.json` or `segment_packet.md`). Rate each segment as a market that a firm could enter with a new kind of offering in the United States, over roughly the last 25 years.

Use only your own knowledge of industries and the segment descriptions. Do not look for, open, or use any other file, dataset, or result.

Score every item from 1 to 5 (integers). Where a segment mixes several kinds of business, rate the typical business in it.

## Industry structure (Porter's five forces, plus scale)

- `entry_threat`: how easy it is for new competitors to enter. 1 = very hard (heavy capital, licences, approvals); 5 = very easy.
- `rivalry`: intensity of competition among existing firms. 1 = mild; 5 = intense.
- `substitutes`: threat from substitute products or services that meet the same need differently. 1 = few; 5 = many close substitutes.
- `buyer_power`: bargaining power of customers and sales channels (large retailers, marketplaces, distributors, procurement departments). 1 = weak; 5 = strong.
- `supplier_power`: bargaining power of suppliers and platforms the business depends on (app stores, marketplaces, cloud hosts, manufacturers, licensors). 1 = weak; 5 = strong.
- `scale_economies`: how strongly scale or network effects let a few firms dominate (winner-take-most). 1 = small firms compete on equal terms; 5 = one or a few firms tend to take most of the market.

## Who profits from a new offering (Teece)

- `imitability`: how easily a new offering in this segment can be copied. 1 = hard to copy (patents, secrets, know-how); 5 = easy to copy.
- `complementary_assets`: how much success depends on specialized assets beyond the idea itself (distribution, manufacturing, regulatory approval, brand, installed base). 1 = little; 5 = a great deal.
- `incumbents_hold_assets`: how far those complementary assets are held by established firms rather than available to newcomers. 1 = available to newcomers; 5 = held by incumbents.

## Overall judgements

- `attractiveness`: long-run profit prospects for a typical entrant. 1 = poor; 5 = good.
- `pioneer_advantage`: when a new kind of offering appears in this segment, who usually ends up with the market? 1 = later entrants or incumbents capture it; 3 = no clear pattern; 5 = the first firms to offer it keep it.

## Output

Write a JSON list to the output path you were given, one object per segment, in this form:

```json
{"segment": "S16", "entry_threat": 4, "rivalry": 4, "substitutes": 3, "buyer_power": 3, "supplier_power": 2,
 "scale_economies": 2, "imitability": 4, "complementary_assets": 2, "incumbents_hold_assets": 2,
 "attractiveness": 2, "pioneer_advantage": 2, "summary": "one sentence naming the business and the main reason for the scores"}
```

Rate all 60 segments. Score each segment on its own; do not rank them against each other.

import { expect, test } from "bun:test";
import { localTargets } from "./check-doc-links";

test("checks inline, reference, image and HTML targets", () => {
  expect(localTargets('[guide](guides/start.md#here)\n![plot](plot.svg)\n[ref]: <a%20b.md> "Title"\n<img src="icon.png">')).toEqual(['guides/start.md', 'plot.svg', 'a b.md', 'icon.png']);
});
test("ignores external links, fragments and fenced examples", () => {
  expect(localTargets('[web](https://example.com) [here](#here) [mail](mailto:a@b)\n```md\n[fake](missing.md)\n```\n[real](real.md)')).toEqual(['real.md']);
});

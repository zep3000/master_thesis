const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const { discoverImageDirOptions, isWithin, resolveImagePath } = require("../server");

test("image folder options use configured local roots", async () => {
  const options = await discoverImageDirOptions();
  assert.ok(options.some((option) => option.label === "Configured image directory 1"));
  assert.equal(options[0].path, "");
});

test("a relative manifest image resolves inside its manifest directory", async (context) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "annotation-comparison-"));
  context.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const images = path.join(root, "images");
  fs.mkdirSync(images);
  const expected = path.join(images, "example-page.jpg");
  fs.writeFileSync(expected, "synthetic image fixture", "utf8");
  const resolved = await resolveImagePath(
    { filename: "example-page.jpg", path: "images/example-page.jpg" },
    root
  );
  assert.equal(resolved, expected);
});

test("path containment rejects sibling-prefix and parent traversal", () => {
  const root = path.resolve("safe-root");
  assert.equal(isWithin(root, path.join(root, "image.jpg")), true);
  assert.equal(isWithin(root, path.resolve(root, "..", "safe-root-copy", "image.jpg")), false);
  assert.equal(isWithin(root, path.resolve(root, "..", "secret.txt")), false);
});

const fs = require("fs");
const path = require("path");

const srcFolder = path.join(__dirname, "tests");
const destFolder = path.join(__dirname, "pkg");

// Copy index file to {workspace}/pkg/ folder
fs.copyFileSync(
  path.join(srcFolder, "index.html"),
  path.join(destFolder, "index.html")
);

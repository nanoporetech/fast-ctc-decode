module.exports = {
  presets: [
    [
      "@babel/preset-env",
      {
        targets: {
          browsers: "last 2 versions, ie 10-11",
        },
        modules: false,
      },
    ],
  ],
  plugins: [
    "@babel/plugin-syntax-dynamic-import",
    "babel-plugin-transform-import-meta",
  ],
  env: {
    test: {
      presets: [
        [
          "@babel/preset-env",
          {
            modules: "auto",
          },
          "jest",
        ],
      ],
      plugins: ["@babel/plugin-transform-runtime"],
    },
  },
};

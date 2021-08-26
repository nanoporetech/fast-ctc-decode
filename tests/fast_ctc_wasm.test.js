const puppeteer = require('puppeteer');

let page = null;
const mockAlphabet = ["N","A","G"];
const mockBeamSize = 5;
const mockBeamCutThreshold = Number(0.0).toPrecision(2);
const mockCollapseRepeats = true;
const mockShape = [10, 3];
const mockString = false;
const mockQbias = Number(0.0).toPrecision(2);
const mockQscale = Number(1.0).toPrecision(2);
const mockFloatArr = [0.0, 0.4, 0.6, 0.0, 0.3, 0.7, 0.3, 0.3, 0.4, 0.4, 0.3, 0.3, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.1, 0.4, 0.5, 0.1, 0.5, 0.4, 0.8, 0.1, 0.1, 0.1, 0.1, 0.8];

describe("Fast CTC Decode Browser", () => {
  beforeEach(async () => {
    browser = await puppeteer.launch({ dumpio: true });
    page = await browser.newPage();
  });

  afterEach(async () => {
    await browser.close();
  })

  it("should setup the correct page", async () => {
    await setup();
    await expect(page.title()).resolves.toMatch("Fast CTC Example");
  });
 
  it("should return correct beam search results", async () => {
    await setup("beam_search", mockFloatArr, [mockAlphabet, mockBeamSize, mockBeamCutThreshold, mockCollapseRepeats, mockShape]);

    const result = await evaluateResult();

    expect(result.seq).toBe("GAGAG");
    expect(result.starts).toEqual([0, 1, 2, 4, 6]);
  });
  
  
  it("should return correct viterbi search results", async () => {
    await setup("viterbi_search", mockFloatArr, [mockAlphabet, mockString, mockQscale, mockQbias, mockCollapseRepeats, mockShape]);

    const result = await evaluateResult();

    expect(result.seq).toBe("GGAG");
    expect(result.starts).toEqual([0, 5, 7, 9]);
  });
});

const evaluateResult = async (divTag = "#result") => {
  const div = await page.$(divTag);
  let result = await page.evaluate((div) => {
    if(div && typeof div.innerHTML === 'string') {
      return JSON.parse(div.innerHTML)
    } else {
      return div.innerHTML;
    }
  }, div);
  result = JSON.parse(result);
  return result;
}

const setup = async (method, posteriors, params) => {
  if (method && posteriors) {
    await windowSet(page, "method", method);
    await windowSet(page, "posteriors", JSON.stringify(posteriors), false);
    params !== undefined && await windowSet(page, "params", JSON.stringify(params), false);
  }
  await page.goto("http://localhost:5000", { waitUntil: "networkidle0" });
};

const windowSet = (page, name, value, isString = true) => {
  page.evaluateOnNewDocument(`
    Object.defineProperty(window, '${name}', {
      get() {
        if(${isString}) return '${value}'
        return ${value}
      }
    })
  `);
};

"use strict";

const byId = id => document.getElementById(id);
const titleCase = value => value ? value[0].toUpperCase() + value.slice(1) : "Unreported";
const number = value => value.toLocaleString("en-US");
const pageSize = 20;
let catalog;
let page = 0;
let filtered = [];
let selectedScope = null;
const scopeCounts = new Map();

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function statusBadge(status) {
  const badge = element("span", status, "state");
  badge.dataset.value = status;
  return badge;
}

function showDialog(title) {
  byId("dialog-title").textContent = title;
  byId("dialog").showModal();
  byId("dialog").scrollTop = 0;
}

function definition(list, key, values) {
  list.append(element("dt", key));
  for (const value of values) list.append(element("dd", value));
}

function showPackage() {
  const list = element("dl");
  definition(list, "Package", [catalog.name]);
  definition(list, "View", [catalog.view]);
  definition(list, "Snapshot commit", [catalog.commit || "Unreported"]);
  definition(list, "Extraction coverage", [titleCase(catalog.coverage)]);
  if (catalog.coverage === "partial") {
    list.append(element("dd", "The extraction may have omitted repository conventions."));
  }
  if (catalog.reasons.length) definition(list, "Coverage notes", catalog.reasons);
  byId("details").replaceChildren(list);
  showDialog("Package details");
}

function browseScopes() {
  const list = element("div", null, "scope-list");
  const entries = [...scopeCounts].sort(([pathA, a], [pathB, b]) => b - a || pathA.localeCompare(pathB, "en"));
  for (const [path, count] of entries) {
    const button = element("button", null, "scope-option");
    const bar = element("progress");
    bar.setAttribute("aria-hidden", "true");
    bar.max = entries[0][1];
    bar.value = count;
    button.append(element("code", path), element("small", number(count) + (count === 1 ? " claim" : " claims")), bar);
    button.addEventListener("click", () => {
      selectedScope = path;
      byId("dialog").close();
      applyFilters();
      byId("clear-scope").focus();
    });
    list.append(button);
  }
  byId("details").replaceChildren(list);
  showDialog("Browse by scope");
}

function showClaim(claim) {
  const body = byId("details");
  body.replaceChildren(statusBadge(claim.status), element("h3", claim.rule),
    element("code", claim.id, "rule-id"));
  const list = element("dl");
  definition(list, "Applies to", claim.scopes);
  const label = claim.status === "Review" ? "Proposed normative strength" : "Normative strength";
  definition(list, label, [titleCase(claim.strength)]);
  for (const [key, values] of Object.entries(claim.fields)) definition(list, key, values);
  body.append(list);
  for (const section of claim.sections) {
    const group = element("section", null, "evidence");
    group.append(element("h4", section.title));
    for (const block of section.blocks) group.append(element("p", block.text, block.kind));
    body.append(group);
  }
  showDialog("Claim details");
}

function renderOverview() {
  byId("package-name").textContent = catalog.name;
  byId("view").textContent = catalog.view;
  byId("coverage").textContent = titleCase(catalog.coverage) + " coverage";
  if (catalog.coverage === "partial") byId("coverage").classList.add("partial");
  byId("total").textContent = number(catalog.claims.length);
  for (const claim of catalog.claims) {
    for (const path of new Set(claim.scopes)) scopeCounts.set(path, (scopeCounts.get(path) || 0) + 1);
  }
  byId("scope-count").textContent = number(scopeCounts.size);
  byId("browse-scopes").disabled = !scopeCounts.size;

  const strengths = new Set(catalog.claims.map(claim => claim.strength));
  for (const strength of new Set(["required", "recommended", "observed", ...strengths])) {
    if (strengths.has(strength)) byId("strength").add(new Option(titleCase(strength), strength));
  }
  for (const status of [...new Set(catalog.claims.map(claim => claim.status))].sort()) {
    byId("status").add(new Option(status, status));
  }
}

function claimRow(claim) {
  const row = element("tr");
  const first = row.insertCell();
  const button = element("button", claim.rule, "rule");
  first.append(button, element("span", claim.id, "rule-id"));
  const scope = row.insertCell();
  scope.append(element("span", claim.scopes[0], "scope"));
  if (claim.scopes.length > 1) scope.append(element("span", "+" + (claim.scopes.length - 1), "additional"));
  const strength = element("span", titleCase(claim.strength), "strength");
  strength.dataset.value = claim.strength;
  row.insertCell().append(strength);
  const status = row.insertCell();
  status.append(statusBadge(claim.status));
  if ([...(claim.fields.Use || []), ...(claim.fields["Delivery policy"] || [])].includes("reference_only")) {
    status.append(element("small", "Reference only", "reference"));
  }
  const arrow = row.insertCell();
  arrow.className = "arrow";
  arrow.setAttribute("aria-hidden", "true");
  arrow.textContent = "›";
  row.addEventListener("click", () => {
    button.focus({preventScroll: true});
    showClaim(claim);
  });
  return row;
}

function renderPage() {
  const start = page * pageSize;
  byId("rows").replaceChildren(...filtered.slice(start, start + pageSize).map(claimRow));
  byId("empty").hidden = filtered.length > 0;
  byId("empty-title").textContent = catalog.claims.length ? "No matching claims" : "This package contains no claims";
  byId("reset").hidden = !catalog.claims.length;
  byId("matching").textContent = number(filtered.length) + (filtered.length === 1 ? " claim" : " claims");
  byId("range").textContent = filtered.length
    ? number(start + 1) + "–" + number(Math.min(start + pageSize, filtered.length)) + " of " + number(filtered.length)
    : "0 results";
  byId("prev").disabled = page === 0;
  byId("next").disabled = start + pageSize >= filtered.length;
}

function applyFilters() {
  const terms = byId("search").value.trim().toLowerCase().split(/\s+/);
  const strength = byId("strength").value;
  const status = byId("status").value;
  filtered = catalog.claims.filter(claim =>
    (!selectedScope || claim.scopes.includes(selectedScope))
    && (!strength || claim.strength === strength) && (!status || claim.status === status)
    && terms.every(term => claim.searchText.includes(term)));
  byId("clear-scope").hidden = !selectedScope;
  byId("clear-scope").textContent = "Scope: " + selectedScope + " ×";
  byId("clear-scope").setAttribute("aria-label", "Clear scope filter " + (selectedScope || ""));
  page = 0;
  renderPage();
}

async function start() {
  const response = await fetch("/package.json");
  if (!response.ok) throw new Error("Could not read package data");
  catalog = await response.json();
  for (const claim of catalog.claims) {
    claim.searchText = [claim.rule, claim.id, ...claim.scopes, ...Object.values(claim.fields).flat(),
      ...claim.sections.flatMap(section => section.blocks.map(block => block.text))].join(" ").toLowerCase();
  }
  renderOverview();
  applyFilters();
  byId("loading").hidden = true;
  byId("app").hidden = false;
}

byId("search").addEventListener("input", applyFilters);
for (const id of ["strength", "status"]) byId(id).addEventListener("change", applyFilters);
byId("reset").addEventListener("click", () => {
  selectedScope = null;
  for (const id of ["search", "strength", "status"]) byId(id).value = "";
  applyFilters();
  byId("search").focus();
});
for (const [id, delta] of [["prev", -1], ["next", 1]]) {
  byId(id).addEventListener("click", () => {
    page += delta;
    renderPage();
    byId("catalog-title").scrollIntoView({block: "start"});
  });
}
byId("browse-scopes").addEventListener("click", browseScopes);
byId("clear-scope").addEventListener("click", () => {
  selectedScope = null;
  applyFilters();
  byId("search").focus();
});
byId("info").addEventListener("click", showPackage);
byId("close").addEventListener("click", () => byId("dialog").close());
start().catch(error => { byId("loading").textContent = "Unable to open the knowledge package. " + error.message; });

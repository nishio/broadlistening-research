# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Sensemaker by Jigsaw** - a proof-of-concept toolkit for analyzing large-scale online conversations using LLMs (primarily Google's Gemini models via Vertex AI). The library provides three core capabilities:

1. **Topic Identification** - Discovers topics, subtopics, and themes in conversation data
2. **Statement Categorization** - Assigns statements to discovered or user-defined topics
3. **Summarization** - Generates narrative reports with vote analysis, common ground, and areas of disagreement

## Repository Structure

This is a **monorepo** using npm workspaces with the following packages:

- **library/** - Core TypeScript library (`sensemaking-tools` package) containing:
  - `src/` - Main library code
  - `runner-cli/` - CLI tools for running the pipeline
  - `evals/` - Evaluation tools for assessing LLM outputs
- **web-ui/** - Angular application for rendering summary reports
- **visualization-library/** - Visualization components
- **api-server/** - API server components
- **napolitan/** - Additional workspace

## Common Commands

### Testing
```bash
# Run all tests across workspaces
npm test

# Run tests for library only (TypeScript)
cd library && npm test

# Run tests in watch mode (for development)
cd library && npm run test-watch

# Run Python tests
cd library && npm run test:py
```

### Linting and Formatting
```bash
# Lint all workspaces
npm run lint

# Format all workspaces
npm run format

# Lint library only
cd library && npm run lint
```

### Documentation
```bash
# Generate TypeDoc documentation (outputs to docs/)
npx typedoc
```

### Building the Web UI
```bash
cd web-ui

# Generate a web report from JSON files (outputs to web-ui/dist/web-ui)
npx ts-node site-build.ts --topics <path> --summary <path> --comments <path> --reportTitle "Title"

# Run development server
npm run dev

# Generate single HTML file for easy sharing
npx ts-node single-html-build.js
```

### Running the CLI Tools

The library provides three CLI runners (all in `library/runner-cli/`):

```bash
# Main runner - generates HTML/Markdown summary
npx ts-node ./library/runner-cli/runner.ts \
  --outputBasename out \
  --vertexProject "YOUR_PROJECT_ID" \
  --inputFile "data.csv"

# Categorization runner - outputs CSV with topic assignments
npx ts-node ./library/runner-cli/categorization_runner.ts \
  --outputBasename out \
  --vertexProject "YOUR_PROJECT_ID" \
  --inputFile "data.csv"

# Advanced runner - outputs JSON files for topics, comments, and summary
npx ts-node ./library/runner-cli/advanced_runner.ts \
  --outputBasename out \
  --vertexProject "YOUR_PROJECT_ID" \
  --inputFile "data.csv"
```

**CSV Input Format:**
- Required columns: `comment-id`, `comment_text`
- Vote data (optional):
  - Without groups: `agrees`, `disagrees`, `passes`
  - With groups: `{group_name}-agree-count`, `{group_name}-disagree-count`, `{group_name}-pass-count`

## Architecture and Key Concepts

### Core Classes and Types

**Sensemaker** (`library/src/sensemaker.ts`)
- Main entry point for the library
- Manages model configuration via `ModelSettings`
- Key methods:
  - `learnTopics()` - Discovers topics from comments
  - `categorizeComments()` - Assigns topics to comments
  - `summarize()` - Generates full summary with vote analysis

**Models** (`library/src/models/`)
- Abstract `Model` base class defines interface for LLM providers
- `VertexModel` - Default implementation using Google Vertex AI
- Custom models can be implemented by extending `Model` class with just two methods:
  - `generateText(prompt: string): Promise<string>`
  - `generateData(prompt: string, schema: TSchema): Promise<T>`

**Type System** (`library/src/types.ts`)
- Uses TypeBox for runtime type validation and JSON Schema generation
- Key types: `Comment`, `Topic`, `Summary`, `VoteTally`, `SummaryContent`
- Topics can be flat (`FlatTopic`) or nested (`NestedTopic` with subtopics)
- Structured types enable Vertex AI's constrained decoding feature

### Task Pipeline

The library processes comments through a pipeline:

1. **Topic Modeling** (`library/src/tasks/topic_modeling.ts`)
   - Can learn 1-3 levels deep (topics → subtopics → themes)
   - Supports user-provided top-level topics

2. **Categorization** (`library/src/tasks/categorization.ts`)
   - Uses `categorizeCommentsRecursive()` for multi-level categorization
   - Processes comments in batches with error handling and retries
   - Leverages Vertex AI's JSON schema validation

3. **Summarization** (`library/src/tasks/summarization.ts`)
   - Supports multiple summarization types via `SummarizationType` enum
   - Subtasks in `library/src/tasks/summarization_subtasks/`:
     - `intro.ts` - Statistics overview
     - `overview.ts` - High-level summary
     - `top_subtopics.ts` - Top 5 subtopics by volume
     - `topics.ts` - Per-topic summaries with themes
     - `groups.ts` - Group-specific analysis
     - `relative_context.ts` - Agreement level analysis
   - Citations link summary claims to source comments

### Vote Analysis

**Statistics** (`library/src/stats/`)
- `summary_stats.ts` - Basic vote aggregation
- `majority_vote.ts` - Simple majority-based analysis
- `group_informed.ts` - Group-aware consensus calculation
- `matrix_factorization.ts` - Advanced opinion clustering using TensorFlow.js

**Common Ground vs. Differences:**
- Statements need ≥20 votes to be included (avoid small sample bias)
- Common ground: High agree rates across groups
- Differences: Mixed agree/disagree patterns
- Relative agreement labels: "high", "moderately high", "moderately low", "low"

### Citation System

Citations ground LLM-generated summaries in source comments:
- XML format: `<citation comment_id=123>`
- MARKDOWN format: `[[123](## "comment text\nVotes: ...")]`
- Implemented in `library/src/tasks/utils/citation_utils.ts`
- Hover tooltips show original comment text and vote counts in HTML output

## Development Guidelines

### Testing Strategy
- Unit tests colocated with source files (e.g., `sensemaker.test.ts`)
- Tests use Jest framework
- Mock data and fixtures used for LLM-dependent code
- Python tests use pytest (`library/evals/` contains Python evaluation code)

### Model Configuration
- Default model: `gemini-2.5-pro` (configurable in `VertexModel` constructor)
- Parallelism controlled via `p-limit` (see `DEFAULT_VERTEX_PARALLELISM` in `model_util.ts`)
- Retry logic built into `callLLM()` with exponential backoff
- Safety settings disabled (set to `BLOCK_NONE`) for civic conversation analysis

### GCloud Authentication
Required for Vertex AI access:
```bash
gcloud config set project <your-project-id>
gcloud auth application-default login
```

### Token Usage and Costs
Approximate token counts for 1000-statement conversations:
- Topic Identification: 130K input / 50K output
- Categorization: 130K input / 50K output
- Summarization: 80K input / 7.5K output
- Total cost (Gemini 1.5 Pro, April 2025): <$1

### Working with Summaries

Summary structure is recursive:
- `Summary.contents` is an array of `SummaryContent`
- Each `SummaryContent` can have `subContents` (for topic/subtopic hierarchy)
- Filter contents with `Summary.withoutContents(removeFn)`
- Export formats: `getText("XML")` or `getText("MARKDOWN")`

### Evaluations
- Hallucination detection in `library/evals/autorating/`
- Silhouette coefficient for topic quality (code to be published)
- Stability analysis shows ~90% consistency in categorization across runs

## Important Notes

- **Recursive Summarization:** The summarization process is hierarchical - understand the recursive flow through topic → subtopic → theme levels
- **Schema Validation:** Vertex AI's constrained decoding relies on TypeBox schemas being correctly defined
- **Categorization is Stateful:** Comments gain a `topics` property after categorization; many functions check if comments are already categorized
- **Vote Thresholds:** The 20-vote minimum for common ground/differences is hardcoded; be aware of this when working with smaller datasets
- **Performance Timing:** Console logs show performance metrics for each major operation
- **Error Handling:** LLM calls include retry logic; check console for validation failures

## Web UI Integration

To display results:
1. Run `advanced_runner.ts` to generate three JSON files (topics, summary, comments)
2. Use `web-ui/site-build.ts` to create Angular build with embedded data
3. Either:
   - Serve with `npm run dev` at localhost:4200
   - Generate single HTML with `single-html-build.js` for easy sharing

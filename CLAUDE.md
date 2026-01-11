# BABEL-BEATS - AI Context Document

## Project Overview
Language learning through AI-generated musical patterns

## Core Concept
- Speak the universal language of music
- AI-powered music generation tailored to specific use case
- Real-time data integration and processing
- User-centric design with clear value proposition

## Technical Architecture
- **Data Sources**: Language databases, Speech patterns, Cultural music libraries
- **AI Models**: 
  - Music generation: Specialized models for use case
  - Data processing: Custom preprocessing pipelines
  - Personalization: User preference learning
- **Frontend**: React/Next.js with appropriate UI/UX
- **Backend**: Python FastAPI, Redis, PostgreSQL
- **Infrastructure**: Cloud-native architecture

## MVP Features
1. Pronunciation rhythm mapping
2. Tonal language training
3. Cultural music integration
4. Basic web interface
5. Sharing and export functionality

## Grant Alignment
- **Innovation**: Novel application of AI in music
- **Impact**: Clear benefit to target audience
- **Scalability**: Cloud-native architecture
- **Sustainability**: Revenue model potential

## Development Status
- [ ] Step 1: Core concept validation
- [ ] Step 2: Technical architecture design
- [ ] Step 3: MVP feature set definition
- [ ] Step 4: Grant alignment mapping
- [ ] Step 5: Initial prototype code

## Key Decisions
- Focus on user value over technical complexity
- Start with core differentiating feature
- Build for scale from day one
- Prioritize user feedback loops

---

# Ralph Loop Development Methodology

## Autonomous Development Protocol

This project uses **Ralph Loop** - a continuous AI agent methodology for iterative development. Named after Ralph Wiggum, it embodies persistent iteration until completion.

### How It Works

1. Claude receives a task with clear completion criteria
2. Works autonomously, iterating until done
3. State persists in files and git, not context
4. Fresh context on each loop prevents pollution
5. Completion signaled via `<promise>COMPLETE</promise>`

### Starting a Ralph Loop

```bash
/ralph-loop "Your task description. Requirements: [list]. Output <promise>COMPLETE</promise> when done." --max-iterations 50
```

### Writing Effective Prompts

**Good prompts have:**
- Binary success criteria (testable pass/fail)
- Incremental phases for complex work
- Self-correction loops (run tests, fix, repeat)
- Escape hatches (`--max-iterations`)

**Example:**
```
Implement user authentication with JWT.

Phase 1: Create auth routes (register, login, logout)
Phase 2: Add JWT token generation and validation
Phase 3: Create auth middleware
Phase 4: Write tests (>80% coverage)
Phase 5: Update README with API docs

Run tests after each phase. Fix any failures before proceeding.
Output <promise>COMPLETE</promise> when all phases pass.
```

---

## MCP Server Configuration

Model Context Protocol (MCP) servers extend Claude Code with specialized capabilities.

### Recommended MCP Servers

| Server | Purpose | Install Command |
|--------|---------|-----------------|
| **Puppeteer** | Browser automation, screenshots, web scraping | `claude mcp add puppeteer -- npx -y @modelcontextprotocol/server-puppeteer` |
| **Supabase** | Database operations, auth, storage | `claude mcp add supabase -- npx -y @supabase/mcp-server-supabase` |
| **GitHub** | Repo management, issues, PRs | `claude mcp add github -- npx -y @anthropic-ai/mcp-github` |
| **Filesystem** | Extended file operations | `claude mcp add filesystem -- npx -y @anthropic-ai/mcp-filesystem` |

### MCP Configuration File

Create `.mcp.json` in project root:

```json
{
  "mcpServers": {
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    },
    "supabase": {
      "command": "npx",
      "args": ["-y", "@supabase/mcp-server-supabase@latest", "--supabase-url", "${SUPABASE_URL}"],
      "env": {
        "SUPABASE_SERVICE_ROLE_KEY": "${SUPABASE_SERVICE_ROLE_KEY}"
      }
    }
  }
}
```

### Puppeteer Capabilities

With Puppeteer MCP, Claude can:
- **Navigate** - Open URLs, click links, fill forms
- **Screenshot** - Capture full page or element screenshots
- **Extract** - Scrape text, attributes, HTML content
- **Execute** - Run JavaScript in browser context
- **Test** - Validate UI behavior, check responsive layouts

**Example prompt:**
```
Use Puppeteer to:
1. Navigate to https://example.com
2. Take a screenshot of the homepage
3. Extract all product names from the catalog
4. Save results to data/products.json
```

---

## GitHub Actions CI

All BlueDuckLLC repos should have CI workflows for automated testing.

### Standard CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Type check
        run: npx tsc --noEmit

      - name: Lint
        run: npm run lint

      - name: Build
        run: npm run build

      - name: Test
        run: npm test --if-present
```

### CI Requirements for Ralph Loop

Before marking a task COMPLETE, verify:
- [ ] `npm run build` passes
- [ ] `npm run lint` passes (if configured)
- [ ] `npm test` passes (if tests exist)
- [ ] No TypeScript errors (`npx tsc --noEmit`)

---

## Long-Form Autonomous Thinking

### When to Think Deeply

Use extended reasoning for:
- Architectural decisions affecting multiple files
- Debugging complex issues across the codebase
- Planning multi-step implementations
- Evaluating trade-offs between approaches

### Subagent Strategy

Launch specialized subagents for parallel work:

| Agent Type | Use Case |
|------------|----------|
| `Explore` | Codebase search, finding patterns |
| `Plan` | Architecture design, implementation strategy |
| `Bash` | Git operations, builds, deployments |
| `general-purpose` | Complex multi-step research |

**Parallel execution example:**
```
When researching a feature, launch:
- Explore agent: Find similar patterns in codebase
- Plan agent: Design implementation approach
- general-purpose agent: Research external docs
```

### Context Management

- State lives in files and git, not LLM memory
- Each loop iteration gets fresh context
- Use @fix_plan.md to track progress across sessions
- Commit frequently to preserve work

---

## Completion Criteria Standards

### For Features
```
Feature is COMPLETE when:
- [ ] All acceptance criteria met
- [ ] Tests pass (coverage >80%)
- [ ] No TypeScript errors
- [ ] Build succeeds
- [ ] CI workflow passes
- [ ] README updated if needed
Output: <promise>COMPLETE</promise>
```

### For Bug Fixes
```
Bug fix is COMPLETE when:
- [ ] Root cause identified and documented
- [ ] Fix implemented
- [ ] Regression test added
- [ ] All existing tests pass
- [ ] CI workflow passes
Output: <promise>COMPLETE</promise>
```

### For Refactoring
```
Refactor is COMPLETE when:
- [ ] Code restructured as specified
- [ ] No behavior changes (tests verify)
- [ ] All tests pass
- [ ] No new TypeScript errors
- [ ] CI workflow passes
Output: <promise>COMPLETE</promise>
```

---

## Ralph Wiggum Technique - Full Implementation

The setup consists of three main components: the Bash script (the orchestrator), the PRD file (the backlog), and the Memory file.

### 1. The Orchestrator: `ralph.sh`
This script runs the "infinite" loop. It executes the CLI coding agent, checks for a stop condition, and manages the iteration count.

```bash
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Usage: ./ralph.sh <max_iterations>
MAX_ITERS=$1

if [ -z "$MAX_ITERS" ]; then
  echo "Please provide the maximum number of iterations."
  exit 1
fi

# The Main Loop
for ((i=1; i<=MAX_ITERS; i++)); do
  echo "--- Iteration $i of $MAX_ITERS ---"

  # Run the CLI Agent (e.g., Claude Code, Open Code)
  # We pass the context files: the PRD and the Progress Log
  # We capture the output to check for the stop condition later
  OUTPUT=$(claude --prompt "
    Read plans/prd.json and progress.txt.
    1. Find the highest priority feature that has 'passes: false'.
    2. Implement ONLY that single feature.
    3. Run type checks (pnpm type check) and tests (pnpm test).
    4. If successful, update plans/prd.json marking the item as 'passes: true'.
    5. Append your learnings/logs to progress.txt.
    6. Git commit the feature, the PRD update, and the progress log.
    7. If ALL items in the PRD are passing, output exactly: 'promise complete here'.
  ")

  # Print the output to the terminal so we can see it working
  echo "$OUTPUT"

  # Check for the specific stop phrase defined in the prompt
  if echo "$OUTPUT" | grep -q "promise complete here"; then
    echo "Ralph has finished the backlog!"
    # Optional: Send a notification (e.g., WhatsApp/Text)
    # npx tsx scripts/notify.ts "Ralph finished after $i iterations"
    exit 0
  fi
done
```

### 2. The Backlog: `plans/prd.json`
This file acts as the project manager. It contains small, distinct user stories. The boolean `passes` flag tells the agent what is done and what is left to do.

```json
[
  {
    "story": "Delete video shows confirmation dialogue before deleting",
    "passes": true
  },
  {
    "story": "Add a 'beat' to a clip: verify that three orange dots appear below the clip",
    "passes": false
  },
  {
    "story": "Verify the beat dots form an ellipses pattern",
    "passes": false
  },
  {
    "story": "Verify the beat dots are orange colored",
    "passes": false
  }
]
```

### 3. The Memory: `progress.txt`
This starts as an empty text file. The agent appends to it after every loop. It serves as context for the next iteration so the agent doesn't repeat mistakes.

```text
(This file starts empty. The agent will append notes like:)
- Iteration 1: Implemented delete confirmation. Note for next sprint: The modal component is located in /src/components/Modal.tsx.
- Iteration 2: Added beat indicator component. Type check failed initially because of missing prop, fixed by adding interface BeatProps.
```

### 4. The "Human-in-the-Loop" Variant: `ralph_once.sh`
This variant is used when you want to steer the agent manually or work on difficult features.

- **Code Difference:** Instead of a `for` loop, it runs the command exactly once using the same prompt structure.
- **Usage:** You run it, watch the output, verify the work, and then decide whether to run it again or intervene manually.

### Key Implementation Details
To make this code actually work, the prompt you feed the LLM must strictly enforce the following rules:

1. **Single Feature Focus:** The agent must be told to work *only* on the highest priority feature to prevent it from consuming too much context or producing low-quality code.
2. **Git Commits:** The agent must commit after every successful loop. This allows the agent to query the git history in future loops to understand what changed.
3. **Feedback Loops:** The agent must be instructed to run tests (e.g., `pnpm test`) inside the loop. If the tests fail, the agent should attempt to fix them within that same iteration before committing.

---

## Anti-Patterns to Avoid

1. **Vague goals** - "Make it better" has no exit condition
2. **Skipping tests** - No verification = no convergence
3. **Unlimited iterations** - Always set `--max-iterations`
4. **Monolithic tasks** - Break into phases
5. **Context pollution** - Let loops refresh context naturally
6. **Ignoring CI** - Always verify CI passes before completion

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `/ralph-loop "..."` | Start autonomous loop |
| `/cancel-ralph` | Stop current loop |
| `--max-iterations N` | Safety limit |
| `--completion-promise "X"` | Custom exit phrase |
| `claude mcp list` | Show configured MCP servers |
| `claude mcp add <name>` | Add MCP server |

---

## Version History

| Date | Update |
|------|--------|
| 2026-01-11 | Added full Ralph Wiggum technique implementation (ralph.sh, prd.json, progress.txt) |
| 2026-01-10 | Added MCP server configuration, CI workflow templates, timestamp tracking |
| 2026-01-10 | Initial Ralph Loop deployment across BlueDuckLLC & LocalEcho repos |

---

*Ralph Loop methodology based on [Geoffrey Huntley's original technique](https://ghuntley.com/ralph/)*
*MCP Puppeteer: [Documentation](https://github.com/anthropics/anthropic-quickstarts/tree/main/mcp-puppeteer)*

**Last updated:** 2026-01-11

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

*Ralph Loop methodology based on [Geoffrey Huntley's original technique](https://ghuntley.com/ralph/)*
*MCP Puppeteer: [Documentation](https://github.com/anthropics/anthropic-quickstarts/tree/main/mcp-puppeteer)*

/**
 * Automated Debug Code Cleanup Script
 * Removes all debugging console.log statements and temporary debug code
 * Designed for production-ready codebase cleanup
 */

const fs = require('fs');
const path = require('path');

class DebugCleanupTool {
  constructor() {
    this.processedFiles = 0;
    this.removedStatements = 0;
    this.debugPatterns = [
      // Console statements with emojis (my debugging style)
      /console\.log\(['"`]🔄[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]📊[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]📋[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]✅[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]❌[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]⚠️[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]🔑[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]📦[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]🚀[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]⏳[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]🏥[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]👁️[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]📑[^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`]📵[^'"`]*['"`][^)]*\);?\s*\n/g,
      
      // Multi-line debug statements
      /console\.log\(\s*['"`][🔄📊📋✅❌⚠️🔑📦🚀⏳🏥👁️📑📵][^'"`]*['"`]\s*[,)][^)]*\);\s*\n/g,
      
      // Specific debugging statements I added
      /console\.log\(['"`].*Starting dashboard.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*User profile.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Session status.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Raw data received.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*API.*result.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Dashboard.*initialized.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Switching from.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Page hidden.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Page visible.*['"`][^)]*\);?\s*\n/g,
      /console\.log\(['"`].*Auto-refreshing.*['"`][^)]*\);?\s*\n/g,
      
      // Generic debugging patterns
      /^\s*\/\/ DEBUG:.*\n/gm,
      /^\s*\/\/ TEMP:.*\n/gm,
      /^\s*\/\/ TESTING:.*\n/gm,
      
      // Console.warn and console.error with emojis
      /console\.warn\(['"`][🔄📊📋✅❌⚠️🔑📦🚀⏳🏥👁️📑📵][^'"`]*['"`][^)]*\);?\s*\n/g,
      /console\.error\(['"`][🔄📊📋✅❌⚠️🔑📦🚀⏳🏥👁️📑📵][^'"`]*['"`][^)]*\);?\s*\n/g,
      
      // Remove empty console.log()
      /^\s*console\.log\(\);\s*\n/gm
    ];
  }

  async cleanupDirectory(dirPath) {
    const items = fs.readdirSync(dirPath);
    
    for (const item of items) {
      const fullPath = path.join(dirPath, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        // Skip node_modules and .next directories
        if (item === 'node_modules' || item === '.next' || item === '.git') {
          continue;
        }
        await this.cleanupDirectory(fullPath);
      } else if (this.shouldProcessFile(fullPath)) {
        await this.cleanupFile(fullPath);
      }
    }
  }

  shouldProcessFile(filePath) {
    const ext = path.extname(filePath);
    const allowedExtensions = ['.js', '.jsx', '.ts', '.tsx'];
    return allowedExtensions.includes(ext);
  }

  async cleanupFile(filePath) {
    try {
      let content = fs.readFileSync(filePath, 'utf8');
      const originalContent = content;
      let removedCount = 0;

      // Apply all debug patterns
      for (const pattern of this.debugPatterns) {
        const matches = content.match(pattern);
        if (matches) {
          removedCount += matches.length;
          content = content.replace(pattern, '');
        }
      }

      // Clean up multiple empty lines (more than 2)
      content = content.replace(/\n{3,}/g, '\n\n');

      // Only write if content changed
      if (content !== originalContent) {
        fs.writeFileSync(filePath, content);
        this.processedFiles++;
        this.removedStatements += removedCount;
        
        console.log(`✅ Cleaned: ${path.relative(process.cwd(), filePath)} (${removedCount} debug statements removed)`);
      }
    } catch (error) {
      console.error(`❌ Error processing ${filePath}:`, error.message);
    }
  }

  async run() {
    console.log('🧹 Starting automated debug cleanup...');
    console.log('📁 Scanning for JavaScript/TypeScript files...');
    
    const startTime = Date.now();
    
    // Clean up main directories
    const dirsToClean = [
      path.join(process.cwd(), 'pages'),
      path.join(process.cwd(), 'components'),
      path.join(process.cwd(), 'lib'),
      path.join(process.cwd(), 'utils')
    ];

    for (const dir of dirsToClean) {
      if (fs.existsSync(dir)) {
        console.log(`🔍 Cleaning directory: ${dir}`);
        await this.cleanupDirectory(dir);
      }
    }

    const duration = Date.now() - startTime;
    
    console.log('\n🎉 Debug cleanup completed!');
    console.log(`📊 Files processed: ${this.processedFiles}`);
    console.log(`🗑️  Debug statements removed: ${this.removedStatements}`);
    console.log(`⏱️  Time taken: ${duration}ms`);
    
    if (this.removedStatements > 0) {
      console.log('\n✨ Codebase is now production-ready!');
    } else {
      console.log('\n✅ No debug statements found - codebase was already clean!');
    }
  }
}

// Run the cleanup if called directly
if (require.main === module) {
  const cleanup = new DebugCleanupTool();
  cleanup.run().catch(console.error);
}

module.exports = DebugCleanupTool;
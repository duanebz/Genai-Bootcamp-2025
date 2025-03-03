import { ThemeProvider } from "@/components/theme-provider"
import { BrowserRouter as Router } from 'react-router-dom'
import AppSidebar from '@/components/Sidebar'
import Breadcrumbs from '@/components/Breadcrumbs'
import AppRouter from '@/components/AppRouter'
import { NavigationProvider } from '@/context/NavigationContext'
import ErrorBoundary from '@/components/ErrorBoundary'
import { SidebarProvider } from "@/components/ui/sidebar"

export default function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <ErrorBoundary>
        <NavigationProvider>
          <Router>
            <SidebarProvider defaultOpen>
              <div className="flex min-h-screen bg-background text-foreground">
                <AppSidebar />
                <main className="flex-1 p-6 overflow-auto">
                  <Breadcrumbs />
                  <AppRouter />
                </main>
              </div>
            </SidebarProvider>  
          </Router>
        </NavigationProvider>
      </ErrorBoundary>
    </ThemeProvider>
  )
}
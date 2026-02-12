"use client"

import * as React from "react"
import { MessageSquare, Plus, Trash, Settings, Menu, X, ChevronLeft } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"

interface ChatSession {
  id: string
  title: string
  created_at: string
}

interface ChatSidebarProps {
  sessions: ChatSession[]
  currentSessionId: string | null
  onSelectSession: (id: string) => void
  onNewChat: () => void
  className?: string
  isOpen: boolean
  onToggle: () => void
}

export function ChatSidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  className,
  isOpen,
  onToggle
}: ChatSidebarProps) {
  
  const SidebarContent = () => (
    <div className="flex h-full flex-col gap-2">
      <div className="flex h-14 items-center px-4 border-b lg:h-[60px]">
        <Button 
            onClick={onNewChat}
            variant="outline" 
            className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
        >
          <Plus size={16} />
          New chat
        </Button>
      </div>
      
      <ScrollArea className="flex-1 px-2">
        <div className="flex flex-col gap-2 py-2">
            <span className="px-2 text-xs font-semibold text-muted-foreground mb-1">Recent</span>
            {sessions.map((session) => (
                <Button
                    key={session.id}
                    variant={currentSessionId === session.id ? "secondary" : "ghost"}
                    className={cn(
                        "justify-start gap-2 h-auto py-3 px-3 text-sm font-normal truncate",
                        currentSessionId === session.id && "bg-secondary"
                    )}
                    onClick={() => onSelectSession(session.id)}
                >
                    <MessageSquare size={16} className="shrink-0" />
                    <span className="truncate text-left w-full">{session.title}</span>
                </Button>
            ))}
            {sessions.length === 0 && (
                <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                    No chats yet. Start a new conversation!
                </div>
            )}
        </div>
      </ScrollArea>
      
      <div className="mt-auto p-4 border-t">
         <div className="flex items-center gap-3 px-2 py-2 text-sm text-muted-foreground">
            <div className="h-2 w-2 rounded-full bg-green-500" />
            <span>Online</span>
         </div>
      </div>
    </div>
  )

  // Mobile Sheet
  if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      // Logic for mobile could be handled by parent or responsive CSS
      // expecting parent to handle layout, this component just renders sidebar content
  }

  return (
    <>
        {/* Desktop Sidebar */}
        <div
            className={cn(
                "hidden border-r bg-muted/10 md:flex md:flex-col transition-all duration-300 ease-in-out",
                isOpen ? "w-64 translate-x-0" : "w-0 -translate-x-full opacity-0 overflow-hidden",
                className
            )}
        >
            <SidebarContent />
        </div>
    </>
  )
}

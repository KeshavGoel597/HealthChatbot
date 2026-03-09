"use client"

import * as React from "react"
import { MessageSquare, Plus, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

interface ChatSession {
  id: string
  title: string
  created_at: string
  expires_at?: string
}

interface ChatSidebarProps {
  sessions: ChatSession[]
  currentSessionId: string | null
  onSelectSession: (id: string) => void
  onNewChat: () => void
  onDeleteSession: (id: string) => void  // GDPR Art. 17 — Right to Erasure
  className?: string
  isOpen: boolean
  onToggle: () => void
}

function getDaysUntilExpiry(expiresAt?: string): number | null {
  if (!expiresAt) return null
  try {
    const expiry = new Date(expiresAt)
    const now = new Date()
    const diffMs = expiry.getTime() - now.getTime()
    return Math.max(0, Math.ceil(diffMs / (1000 * 60 * 60 * 24)))
  } catch {
    return null
  }
}

export function ChatSidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  className,
  isOpen,
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
        <div className="flex flex-col gap-1 py-2">
          <span className="px-2 text-xs font-semibold text-muted-foreground mb-1">Recent</span>
          {sessions.map((session) => {
            const daysLeft = getDaysUntilExpiry(session.expires_at)
            const isExpiringSoon = daysLeft !== null && daysLeft <= 3

            return (
              <div
                key={session.id}
                className={cn(
                  "group flex items-center gap-1 rounded-md px-1",
                  currentSessionId === session.id && "bg-secondary"
                )}
              >
                <Button
                  variant="ghost"
                  className={cn(
                    "flex-1 justify-start gap-2 h-auto py-2.5 px-2 text-sm font-normal min-w-0",
                    currentSessionId === session.id && "bg-secondary hover:bg-secondary"
                  )}
                  onClick={() => onSelectSession(session.id)}
                >
                  <MessageSquare size={14} className="shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1 text-left">
                    <span className="truncate block">{session.title}</span>
                    {/* GDPR Art. 5(1)(e) — Show retention expiry */}
                    {daysLeft !== null && (
                      <span className={cn(
                        "text-[9px] block truncate",
                        isExpiringSoon ? "text-amber-500" : "text-muted-foreground/50"
                      )}>
                        {daysLeft === 0 ? "Expires today" : `Expires in ${daysLeft}d`}
                      </span>
                    )}
                  </div>
                </Button>

                {/* GDPR Art. 17 — Right to Erasure: delete button per session */}
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0 text-muted-foreground/40 hover:text-destructive hover:bg-destructive/10 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation()
                    if (confirm("Delete this chat? This action cannot be undone (GDPR Art. 17 — Right to Erasure).")) {
                      onDeleteSession(session.id)
                    }
                  }}
                  title="Delete this chat (GDPR Art. 17)"
                >
                  <Trash2 size={12} />
                </Button>
              </div>
            )
          })}
          {sessions.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              No chats yet. Start a new conversation!
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="mt-auto p-4 border-t space-y-2">
        <div className="flex items-center gap-3 px-2 py-2 text-sm text-muted-foreground">
          <div className="h-2 w-2 rounded-full bg-green-500" />
          <span>Online</span>
        </div>
        <p className="text-[9px] text-muted-foreground/50 px-2 leading-relaxed">
          Chats auto-deleted after 30 days (GDPR Art. 5(1)(e)).
          Delete any time via 🗑 icon.
        </p>
      </div>
    </div>
  )

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

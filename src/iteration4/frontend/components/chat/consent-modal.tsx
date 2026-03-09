'use client'

import * as React from "react"
import { useState } from "react"
import { Shield, AlertTriangle, CheckCircle2, Info } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ConsentModalProps {
  onConsent: (emrConsent: boolean, storeHistoryConsent: boolean) => void
}

export function ConsentModal({ onConsent }: ConsentModalProps) {
  const [emrConsent, setEmrConsent] = useState(false)
  const [storeHistoryConsent, setStoreHistoryConsent] = useState(false)

  const handleContinue = () => {
    onConsent(emrConsent, storeHistoryConsent)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-sm">
      <div className="w-full max-w-lg mx-4 rounded-2xl border bg-card shadow-2xl overflow-hidden">
        
        {/* Header */}
        <div className="bg-primary/5 border-b px-6 py-5 flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
            <Shield className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-base font-semibold">Data Privacy Notice</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              GDPR Compliance — Your rights matter
            </p>
          </div>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-5">

          {/* AI Disclaimer — GDPR Art. 5(1)(a) + Art. 22 */}
          <div className="flex gap-3 rounded-lg bg-amber-500/10 border border-amber-500/20 px-4 py-3">
            <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
            <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed">
              <strong>Robert is an AI assistant,</strong> not a licensed physician. All information
              provided is for educational purposes only and must be verified by a qualified
              healthcare professional before acting on it.
            </p>
          </div>

          {/* Consent options */}
          <div className="space-y-3">
            <p className="text-sm font-medium">Please choose your data preferences:</p>

            {/* EMR Consent — GDPR Art. 5(1)(a) */}
            <label
              htmlFor="emr-consent"
              className={`flex items-start gap-3 rounded-xl border p-4 cursor-pointer transition-colors ${
                emrConsent
                  ? "border-primary/50 bg-primary/5"
                  : "border-border hover:border-muted-foreground/40"
              }`}
            >
              <input
                id="emr-consent"
                type="checkbox"
                checked={emrConsent}
                onChange={(e) => setEmrConsent(e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-primary shrink-0"
              />
              <div>
                <p className="text-sm font-medium">Allow access to my Electronic Medical Records (EMR)</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Robert will read your diagnoses, medications, symptoms, and lab results
                  to personalise responses. Data is accessed <strong>read-only</strong> and
                  only during your active session.{" "}
                  <span className="text-primary">If unchecked, Robert will only provide general guidance.</span>
                </p>
                <p className="text-[10px] text-muted-foreground/70 mt-1.5">GDPR Art. 5(1)(a) — Lawfulness &amp; Transparency</p>
              </div>
            </label>

            {/* History Storage Consent — GDPR Art. 5(1)(e) */}
            <label
              htmlFor="history-consent"
              className={`flex items-start gap-3 rounded-xl border p-4 cursor-pointer transition-colors ${
                storeHistoryConsent
                  ? "border-primary/50 bg-primary/5"
                  : "border-border hover:border-muted-foreground/40"
              }`}
            >
              <input
                id="history-consent"
                type="checkbox"
                checked={storeHistoryConsent}
                onChange={(e) => setStoreHistoryConsent(e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-primary shrink-0"
              />
              <div>
                <p className="text-sm font-medium">Store this conversation for future reference</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Your chat history will be saved for up to <strong>30 days</strong>,
                  then automatically deleted. You can erase it at any time.
                  Only conversation text is stored — <span className="text-primary">no raw EMR data</span> is ever
                  saved in chat logs.{" "}
                  <span className="text-muted-foreground/70">If unchecked, chats are ephemeral and lost on refresh.</span>
                </p>
                <p className="text-[10px] text-muted-foreground/70 mt-1.5">GDPR Art. 5(1)(e) — Storage Limitation</p>
              </div>
            </label>
          </div>

          {/* Data rights notice */}
          <div className="flex gap-2.5 rounded-lg bg-muted/50 border px-4 py-3">
            <Info className="h-3.5 w-3.5 text-muted-foreground mt-0.5 shrink-0" />
            <p className="text-xs text-muted-foreground leading-relaxed">
              You can delete your stored conversations at any time from the sidebar (🗑).
              EMR records can only be corrected through your healthcare provider — Robert
              cannot modify them. <span className="font-medium">No patient data is used 
              for AI training.</span>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 pb-6">
          <Button
            onClick={handleContinue}
            className="w-full gap-2"
            size="lg"
          >
            <CheckCircle2 className="h-4 w-4" />
            {emrConsent || storeHistoryConsent ? "Save preferences and continue" : "Continue without data sharing"}
          </Button>
          <p className="text-[10px] text-center text-muted-foreground mt-3">
            You can update these preferences at any time by starting a new conversation.
            GDPR Art. 7 — Consent may be withdrawn at any time.
          </p>
        </div>
      </div>
    </div>
  )
}
